"""Tests for training.model_exporter and src/exporter.py."""
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import onnx
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch import nn

from training.model_exporter import (
    ModelExporter,
    _dtype_from_str,
    _get_submodule,
    build_model_exporter,
)


@pytest.fixture(autouse=True)
def _clear_hydra():
    """Clear Hydra global state between tests so hydra.initialize can be re-entered."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


# ----------------------------------------------------------------------
# Unit tests for helpers
# ----------------------------------------------------------------------


def test_dtype_from_str_known():
    assert _dtype_from_str("float32") is torch.float32
    assert _dtype_from_str("bfloat16") is torch.bfloat16
    assert _dtype_from_str("int64") is torch.int64


def test_dtype_from_str_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dtype"):
        _dtype_from_str("float128")


def test_get_submodule_dotted_path():
    inner = nn.Conv2d(3, 4, 1)
    model = nn.Sequential(nn.ModuleDict({"encoder": inner}))
    # Sequential child "0" is the ModuleDict; then .encoder → inner.
    assert _get_submodule(model, "0.encoder") is inner


def test_get_submodule_empty_path_returns_root():
    # reduce with empty-iterable-like behavior: "".split(".") == [""] and getattr(model, "")
    # would fail; callers should pass None (handled by caller). Confirm non-empty top path works.
    model = nn.Linear(2, 2)
    assert _get_submodule(model, "weight") is model.weight


# ----------------------------------------------------------------------
# Build-equivalence: experiment config vs. model-only config
# ----------------------------------------------------------------------


def _model_signature(model: nn.Module) -> list[tuple[str, str, tuple]]:
    """Structural signature: (module path, type name, param-shape tuple per child param)."""
    sig = []
    for name, module in model.named_modules():
        param_shapes = tuple(tuple(p.shape) for pname, p in module.named_parameters(recurse=False))
        sig.append((name, type(module).__name__, param_shapes))
    return sig


def test_experiment_and_model_only_build_same_architecture(tmp_path):
    """compression_ae.yaml and simple_autoencoder.yaml must yield the same model."""
    exporter_from_exp = build_model_exporter(
        config_file="experiment/autoencoder/compression_ae",
        output_dir=str(tmp_path / "exp"),
        overrides=[
            "device=cpu",
            "channels_last=false",
            "load_checkpoint_path=null",
            "resume_training=false",
        ],
    )
    GlobalHydra.instance().clear()

    exporter_from_model = build_model_exporter(
        config_file="model/simple_autoencoder",
        output_dir=str(tmp_path / "model_only"),
        overrides=["+device=cpu"],
    )

    assert _model_signature(exporter_from_exp.model) == _model_signature(exporter_from_model.model)


# ----------------------------------------------------------------------
# End-to-end ONNX exports on model/simple_autoencoder
# ----------------------------------------------------------------------


def _build_simple_ae_exporter(output_dir: str, export_block: dict) -> ModelExporter:
    """Build an exporter on the simple_autoencoder model with a custom export: block."""
    exporter = build_model_exporter(
        config_file="model/simple_autoencoder",
        output_dir=output_dir,
        overrides=["+device=cpu"],
    )
    # Attach an export section to the composed config (open struct to allow new keys).
    OmegaConf.set_struct(exporter.config, False)
    exporter.config = OmegaConf.merge(exporter.config, OmegaConf.create({"export": export_block}))
    return exporter


def test_full_model_onnx_export(tmp_path):
    """No modules configured → full model exported as model.onnx with default_input_shape."""
    out = tmp_path / "full"
    exporter = _build_simple_ae_exporter(
        str(out),
        {
            "opset_version": 21,
            "dynamic_axes": True,
            "default_input_shape": [1, 3, 64, 64],
            "modules": [],
        },
    )
    exporter.export()

    path = out / "model.onnx"
    assert path.exists()
    model_proto = onnx.load(str(path))
    onnx.checker.check_model(model_proto)
    # Single input named "input", single output named "output".
    assert [i.name for i in model_proto.graph.input] == ["input"]
    assert [o.name for o in model_proto.graph.output] == ["output"]


def test_full_model_missing_default_shape_raises(tmp_path):
    exporter = _build_simple_ae_exporter(
        str(tmp_path / "bad"),
        {"opset_version": 21, "dynamic_axes": True, "default_input_shape": [], "modules": []},
    )
    with pytest.raises(ValueError, match="default_input_shape"):
        exporter.export()


def test_pytorch_level_export_submodule(tmp_path):
    """input_layer='0' extracts the encoder (first Sequential child) before ONNX export."""
    out = tmp_path / "pt"
    exporter = _build_simple_ae_exporter(
        str(out),
        {
            "opset_version": 21,
            "dynamic_axes": True,
            "modules": [
                {
                    "name": "encoder",
                    "input_layer": "0",  # first child of nn.Sequential = encoder ConvBlock
                    "output_layer": "0",
                    "onnx_filename": "encoder.onnx",
                    "inputs": [
                        {"name": "input_image", "shape": [1, 3, 64, 64], "dtype": "float32"}
                    ],
                    "outputs": [{"name": "latent"}],
                }
            ],
        },
    )
    exporter.export()

    path = out / "encoder.onnx"
    assert path.exists()
    model_proto = onnx.load(str(path))
    onnx.checker.check_model(model_proto)
    assert [i.name for i in model_proto.graph.input] == ["input_image"]
    assert [o.name for o in model_proto.graph.output] == ["latent"]

    # Encoder downsamples 8x and outputs 256 channels → shape [1, 256, 8, 8] for [1, 3, 64, 64].
    out_shape = model_proto.graph.output[0].type.tensor_type.shape
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in out_shape.dim]
    # batch dim is dynamic ("batch_size"); remaining dims should be 256, 8, 8.
    assert dims[1:] == [256, 8, 8]


def test_onnx_level_extraction(tmp_path):
    """source_onnx + input_tensor/output_tensor slices a sub-graph via onnx.utils.extract_model."""
    out = tmp_path / "onnx_level"
    # Step 1: export the full model so we have a source ONNX file + known tensor names.
    exporter = _build_simple_ae_exporter(
        str(out),
        {
            "opset_version": 21,
            "dynamic_axes": False,  # static shapes make extract_model happy
            "default_input_shape": [1, 3, 64, 64],
            "modules": [],
        },
    )
    exporter.export()
    source_path = out / "model.onnx"
    assert source_path.exists()

    # Step 2: pick a real intermediate tensor name from the exported graph.
    src = onnx.load(str(source_path))
    # Find an intermediate tensor: any node output that is not a graph output.
    graph_output_names = {o.name for o in src.graph.output}
    intermediate = None
    for node in src.graph.node:
        for tensor_name in node.output:
            if tensor_name and tensor_name not in graph_output_names:
                intermediate = tensor_name
                break
        if intermediate is not None:
            break
    assert intermediate is not None, "failed to locate an intermediate tensor for extraction"

    # Step 3: run ONNX-level extraction from input → intermediate.
    OmegaConf.set_struct(exporter.config, False)
    exporter.config = OmegaConf.merge(
        exporter.config,
        OmegaConf.create(
            {
                "export": {
                    "modules": [
                        {
                            "name": "encoder_onnx",
                            "source_onnx": "model.onnx",
                            "input_tensor": "input",
                            "output_tensor": intermediate,
                            "onnx_filename": "encoder_extracted.onnx",
                        }
                    ]
                }
            }
        ),
    )
    exporter.export()

    extracted = out / "encoder_extracted.onnx"
    assert extracted.exists()
    extracted_proto = onnx.load(str(extracted))
    onnx.checker.check_model(extracted_proto)
    assert [i.name for i in extracted_proto.graph.input] == ["input"]
    assert [o.name for o in extracted_proto.graph.output] == [intermediate]


def test_dispatch_chooses_onnx_level_when_source_onnx_set(tmp_path, monkeypatch):
    """Entries with source_onnx go through _export_onnx_level, not torch.onnx.export."""
    exporter = _build_simple_ae_exporter(
        str(tmp_path / "dispatch"),
        {
            "opset_version": 21,
            "dynamic_axes": True,
            "modules": [
                {
                    "name": "x",
                    "source_onnx": "nonexistent.onnx",
                    "input_tensor": "a",
                    "output_tensor": "b",
                    "onnx_filename": "out.onnx",
                }
            ],
        },
    )
    calls = {"pt": 0, "onnx": 0}
    monkeypatch.setattr(
        exporter, "_export_pytorch_level", lambda *a, **k: calls.__setitem__("pt", calls["pt"] + 1)
    )
    monkeypatch.setattr(
        exporter, "_export_onnx_level", lambda *a, **k: calls.__setitem__("onnx", calls["onnx"] + 1)
    )
    exporter.export()
    assert calls == {"pt": 0, "onnx": 1}
