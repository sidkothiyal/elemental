"""Post-processor pipeline: step definitions and orchestration."""

import logging
from typing import Any

from omegaconf import DictConfig

from postprocessor.base import BasePostProcessor

logger = logging.getLogger(__name__)


def extract_value(source: Any, key: str | None) -> Any:
    """Extract a value from *source* by key, handling dicts and objects.

    Args:
        source: A dict, object, or scalar to extract from.
        key: Attribute / dict key to look up.  ``None`` returns *source* as-is.

    Returns:
        The extracted value.
    """
    if key is None:
        return source
    if isinstance(source, dict):
        return source[key]
    return getattr(source, key)


class PostProcessorStep:  # pylint: disable=too-few-public-methods
    """A single step in a post-processor pipeline.

    Each step wraps a :class:`BasePostProcessor` together with metadata that
    tells the pipeline how to feed inputs and where to store outputs.

    Args:
        processor: The post-processor instance to call.
        input_mapping: Maps *data paths* to keyword-argument names for the
            processor.  A data path is a key in the unified data container
            (e.g. ``"pred"``, ``"input/image"``, ``"post_processed/x"``).
            The special path ``"input"`` (without a sub-key) collects every
            ``input/*`` entry into a sub-dict.
        output_key: Key under which the result is stored in the data
            container (e.g. ``"post_processed/formatted_pred"``).
        unfurl_output: If ``True`` **and** the processor returns a ``dict``,
            each key-value pair is merged directly into the data container
            instead of nesting the whole dict under *output_key*.
        phases: Optional list of phases (``"train"``/``"val"``/``"test"``) the
            step runs in. ``None`` (default) runs every phase. Use ``["val"]`` to
            confine an expensive step (e.g. autoregressive generation) to
            validation instead of every training micro-batch.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        processor: BasePostProcessor,
        input_mapping: dict[str, str],
        output_key: str = "post_processed/output",
        unfurl_output: bool = False,
        phases: list[str] | None = None,
    ):
        self.processor = processor
        self.input_mapping = input_mapping
        self.output_key = output_key
        self.unfurl_output = unfurl_output
        self.phases = list(phases) if phases is not None else None


class PostProcessorPipeline:  # pylint: disable=too-few-public-methods
    """Ordered sequence of :class:`PostProcessorStep` instances.

    The pipeline runs each step in order against a shared *data* dict,
    writing results back into the same dict so later steps can consume
    earlier outputs.

    Args:
        steps: Ordered list or dict of :class:`PostProcessorStep` instances.
            If a dict, the values are extracted in insertion order.
    """

    INPUT_PREFIX = "input/"

    def __init__(
        self,
        steps: list[PostProcessorStep] | dict[str, PostProcessorStep] | DictConfig,
    ):
        if isinstance(steps, (dict, DictConfig)):
            steps = list(steps.values())
        else:
            steps = list(steps)

        self.steps = steps
        logger.info("PostProcessorPipeline initialized with %d step(s)", len(self.steps))

    def _resolve_input(self, path: str, data: dict) -> Any:
        """Resolve a single input-mapping path to its value.

        Supported path formats:
        * ``"input"``         -- collect all ``input/*`` entries into a
          sub-dict with the prefix stripped.
        * any other string    -- direct key lookup in *data*.
        """
        if path == "input":
            return {
                k[len(self.INPUT_PREFIX) :]: v
                for k, v in data.items()
                if k.startswith(self.INPUT_PREFIX)
            }

        value = data.get(path)
        if value is not None:
            return value

        # Fallback: try treating the value stored under the path as a nested
        # source (dict / object) and extract a sub-key.
        if "/" in path:
            prefix, subkey = path.split("/", maxsplit=1)
            parent = data.get(prefix)
            if parent is not None:
                return extract_value(parent, subkey)

        return None

    def __call__(self, data: dict) -> dict:
        """Run every step, mutating *data* in place.

        Args:
            data: The unified data container built by
                :func:`build_data_container`.

        Returns:
            The same *data* dict, now augmented with post-processor outputs.
        """
        phase = data.get("phase")
        for step in self.steps:
            # Phase gate: skip steps not enabled for the current phase (e.g.
            # generation confined to ``["val"]`` so it never runs per train batch).
            if step.phases is not None and phase is not None and phase not in step.phases:
                continue
            kwargs: dict[str, Any] = {}
            for path, kwarg_name in step.input_mapping.items():
                resolved = self._resolve_input(path, data)
                if resolved is None:
                    raise KeyError(
                        f"PostProcessorPipeline: could not resolve path {path!r} "
                        f"for step with output_key={step.output_key!r}. "
                        f"Available data keys: {sorted(data.keys())}"
                    )
                kwargs[kwarg_name] = resolved

            result = step.processor(**kwargs)

            if step.unfurl_output and isinstance(result, dict):
                data.update(result)
            else:
                data[step.output_key] = result

        return data
