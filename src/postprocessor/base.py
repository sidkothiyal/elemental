"""Base class for post-processors."""

from abc import ABC, abstractmethod
from typing import Any


class BasePostProcessor(ABC):
    """Abstract base class for all post-processors.

    Subclasses must implement ``__call__`` which receives keyword arguments
    determined by the ``input_mapping`` of the enclosing
    :class:`PostProcessorStep` and returns a result that is stored back
    into the unified data container.
    """

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """Run the post-processor.

        Args:
            **kwargs: Named inputs extracted from the data container
                according to the step's ``input_mapping``.

        Returns:
            Processed result.  If the step's ``unfurl_output`` is True
            and the return value is a dict, each key-value pair is merged
            into the data container; otherwise the whole return value is
            stored under the step's ``output_key``.
        """
