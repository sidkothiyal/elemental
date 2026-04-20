"""Model export utilities for ONNX export and sub-model extraction.

Supports two config sources:
- Full experiment config (e.g. ``experiment/autoencoder/compression_ae``)
- Model-only config    (e.g. ``model/vq_vae_transformer``)

And two extraction methods per module entry:
- PyTorch-level: ``input_layer`` / ``output_layer`` (dotted module paths resolved
  before torch.onnx.export, e.g. ``"model.encoder"``)
- ONNX-level: ``source_onnx`` + ``input_tensor`` / ``output_tensor`` (tensor names
  in an already-exported graph, sliced via ``onnx.utils.extract_model``)
"""

import logging
import os
from functools import reduce
from typing import Any

import hydra
import onnx
import torch
from onnx.utils import extract_model
from torch import nn
from torch.export import Dim

from training.trainer import Trainer
from utils import initializer

logger = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
}


def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted module path (e.g. ``'model.encoder'``) to a submodule."""
    return reduce(getattr, path.split("."), model)


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Valid: {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[dtype_str]


class ModelExporter:
    """Exports PyTorch models to ONNX, with optional sub-model extraction.

    Each entry in ``config.export.modules`` is exported independently.
    PyTorch-level entries use ``input_layer`` / ``output_layer`` to extract a
    submodule before calling ``torch.onnx.export``.  ONNX-level entries use
    ``source_onnx`` + ``input_tensor`` / ``output_tensor`` and delegate to
    ``onnx.utils.extract_model`` to slice an already-exported ONNX graph.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        config: Any,
        device: torch.device,
        output_dir: str,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def export(self) -> None:
        """Export all modules listed in ``config.export.modules``.

        If no modules are configured, exports the full model as ``model.onnx``
        using a single ``input`` tensor whose shape must be supplied via
        ``export.default_input_shape`` (list of ints).
        """
        export_cfg = self.config.get("export", {})
        modules = list(export_cfg.get("modules", []) or [])
        opset_version = export_cfg.get("opset_version", 17)
        dynamic_axes = export_cfg.get("dynamic_axes", True)

        self.model.eval()

        if not modules:
            self._export_full_model_default(export_cfg, opset_version, dynamic_axes)
            return

        for module_cfg in modules:
            name = module_cfg.get("name", "unnamed")
            logger.info("Exporting module: %s", name)
            if module_cfg.get("source_onnx") is not None:
                self._export_onnx_level(module_cfg)
            else:
                self._export_pytorch_level(module_cfg, opset_version, dynamic_axes)
            logger.info("Done exporting module: %s", name)

    # ------------------------------------------------------------------
    # Private: dispatch
    # ------------------------------------------------------------------

    def _export_full_model_default(
        self, export_cfg: Any, opset_version: int, dynamic_axes: bool
    ) -> None:
        """Fallback when no modules are configured — export the whole model."""
        shape = list(export_cfg.get("default_input_shape", []))
        if not shape:
            raise ValueError(
                "export.default_input_shape must be set when no modules are configured"
            )
        inputs = [{"name": "input", "shape": shape, "dtype": "float32"}]
        outputs = [{"name": "output"}]
        self._run_onnx_export(
            module=self.model,
            inputs=inputs,
            outputs=outputs,
            onnx_filename="model.onnx",
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )

    def _export_pytorch_level(
        self, module_cfg: Any, opset_version: int, dynamic_axes: bool
    ) -> None:
        """PyTorch-level extraction: resolve module path then call torch.onnx.export."""
        input_layer = module_cfg.get("input_layer")
        module = _get_submodule(self.model, input_layer) if input_layer else self.model
        self._run_onnx_export(
            module=module,
            inputs=list(module_cfg.get("inputs", [])),
            outputs=list(module_cfg.get("outputs", [])),
            onnx_filename=module_cfg["onnx_filename"],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )

    def _export_onnx_level(self, module_cfg: Any) -> None:
        """ONNX-level extraction via ``onnx.utils.extract_model``.

        Optional ``input_name`` / ``output_name`` (string or list) rename the
        boundary tensors in the extracted graph — useful when chaining sub-models
        (e.g. encoder output renamed to match decoder input).
        """
        source_path = os.path.join(self.output_dir, module_cfg["source_onnx"])
        output_path = os.path.join(self.output_dir, module_cfg["onnx_filename"])

        input_tensor = module_cfg["input_tensor"]
        output_tensor = module_cfg["output_tensor"]
        input_tensors = [input_tensor] if isinstance(input_tensor, str) else list(input_tensor)
        output_tensors = [output_tensor] if isinstance(output_tensor, str) else list(output_tensor)

        logger.info(
            "Extracting ONNX sub-model from %s  inputs=%s  outputs=%s",
            source_path,
            input_tensors,
            output_tensors,
        )
        extract_model(source_path, output_path, input_tensors, output_tensors)

        new_input_names = module_cfg.get("input_name")
        new_output_names = module_cfg.get("output_name")
        if new_input_names is not None or new_output_names is not None:
            self._rename_onnx_io(
                onnx_path=output_path,
                old_inputs=input_tensors,
                new_inputs=new_input_names,
                old_outputs=output_tensors,
                new_outputs=new_output_names,
            )

        logger.info("ONNX sub-model written to: %s", output_path)

    @staticmethod
    def _rename_onnx_io(
        *,
        onnx_path: str,
        old_inputs: list[str],
        new_inputs: str | list[str] | None,
        old_outputs: list[str],
        new_outputs: str | list[str] | None,
    ) -> None:
        """Rename input/output tensors in an ONNX model in-place.

        Replaces every occurrence of the old tensor names across graph inputs,
        graph outputs, and node I/O references.
        """

        def _as_list(value: str | list[str] | None) -> list[str]:
            if value is None:
                return []
            return [value] if isinstance(value, str) else list(value)

        new_input_list = _as_list(new_inputs)
        new_output_list = _as_list(new_outputs)

        rename_map: dict[str, str] = {}
        if new_input_list:
            if len(new_input_list) != len(old_inputs):
                raise ValueError(
                    f"input_name count ({len(new_input_list)}) must match "
                    f"input_tensor count ({len(old_inputs)})"
                )
            rename_map.update(zip(old_inputs, new_input_list))
        if new_output_list:
            if len(new_output_list) != len(old_outputs):
                raise ValueError(
                    f"output_name count ({len(new_output_list)}) must match "
                    f"output_tensor count ({len(old_outputs)})"
                )
            rename_map.update(zip(old_outputs, new_output_list))

        if not rename_map:
            return

        model = onnx.load(onnx_path)
        for tensor in list(model.graph.input) + list(model.graph.output):
            if tensor.name in rename_map:
                tensor.name = rename_map[tensor.name]
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name in rename_map:
                    node.input[i] = rename_map[name]
            for i, name in enumerate(node.output):
                if name in rename_map:
                    node.output[i] = rename_map[name]
        onnx.save(model, onnx_path)
        logger.info("Renamed ONNX I/O tensors: %s", rename_map)

    # ------------------------------------------------------------------
    # Private: ONNX export helpers
    # ------------------------------------------------------------------

    def _build_dummy_inputs(self, inputs: list) -> tuple[torch.Tensor, ...]:
        tensors = []
        for inp in inputs:
            shape = inp.get("shape")
            if not shape:
                raise ValueError(f"Input '{inp.get('name')}' requires a non-empty 'shape'")
            dtype = _dtype_from_str(inp.get("dtype", "float32"))
            tensors.append(torch.zeros(tuple(shape), dtype=dtype, device=self.device))
        return tuple(tensors)

    def _run_onnx_export(
        self,
        *,
        module: nn.Module,
        inputs: list,
        outputs: list,
        onnx_filename: str,
        opset_version: int,
        dynamic_axes: bool,
    ) -> None:
        output_path = os.path.join(self.output_dir, onnx_filename)
        dummy_inputs = self._build_dummy_inputs(inputs)
        input_names = [inp["name"] for inp in inputs]
        output_names = [out["name"] for out in outputs]

        # dynamic_shapes uses torch.export.Dim — one constraint dict per input tensor,
        # matching the positional args tuple. Each input spec may declare a
        # `dynamic_axes: {axis: symbolic_name}` map to mark additional dims dynamic
        # (e.g. H/W for a fully-conv model); otherwise only batch (dim 0) is symbolic.
        shapes: tuple | None = None
        if dynamic_axes:
            dim_cache: dict[str, Dim] = {}

            def _get_dim(name: str) -> Dim:
                if name not in dim_cache:
                    dim_cache[name] = Dim(name)
                return dim_cache[name]

            per_input_shapes = []
            for inp in inputs:
                axes = inp.get("dynamic_axes") or {0: "batch_size"}
                per_input_shapes.append({int(axis): _get_dim(sym) for axis, sym in axes.items()})
            shapes = tuple(per_input_shapes)

        logger.info("torch.onnx.export → %s", output_path)
        torch.onnx.export(
            module,
            dummy_inputs,
            output_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=shapes,
        )
        logger.info("ONNX model written to: %s", output_path)


# ------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------


def build_model_exporter(
    config_file: str,
    output_dir: str,
    checkpoint: str | None = None,
    overrides: list[str] | None = None,
) -> ModelExporter:
    """Build a :class:`ModelExporter` from a config file.

    ``config_file`` may be:

    - A full experiment config (e.g. ``"experiment/autoencoder/compression_ae"``)
      — the ``model:`` block, device, dtype, and checkpoint settings are read
      from the composed config exactly as in training.
    - A model-only config (e.g. ``"model/vq_vae_transformer"``) — minimal config
      containing only ``model:``; device defaults to ``"gpu"`` unless overridden.
    - A dedicated export config (e.g. ``"export/vq_vae_transformer"``) — composes
      a model config via ``defaults:`` and adds an ``export:`` section.

    Args:
        config_file: Hydra config name relative to ``configs/``.
        output_dir: Directory where ONNX files are written.
        checkpoint: Path to checkpoint file (overrides ``load_checkpoint_path``
            in the config when provided).
        overrides: Additional Hydra overrides.
    """
    all_overrides = list(overrides or [])
    if checkpoint is not None:
        all_overrides.append(f"load_checkpoint_path={checkpoint}")

    with hydra.initialize(version_base=None, config_path="../../configs"):
        config = hydra.compose(config_name=config_file, overrides=all_overrides)

    device = initializer.initialize_device(config.get("device", "gpu"))
    dtype = initializer.initialize_dtype(config.get("dtype"))

    model = Trainer._build_model(config, device, use_distributed=False)

    checkpoint_path = config.get("load_checkpoint_path")
    if checkpoint_path is not None:
        initializer.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            criterion=None,
            device=device,
            load_keys=config.get("load_keys"),
            resume_training=False,
            strict=config.get("strict", True),
            results_dir=config.get("results_dir"),
            experiment_name=config.get("experiment_name"),
        )

    model.eval()

    logger.info(
        "ModelExporter ready — device=%s  dtype=%s  output_dir=%s",
        device,
        dtype,
        output_dir,
    )
    return ModelExporter(model=model, config=config, device=device, output_dir=output_dir)
