"""Callback class for Trainer."""


class Callback:
    """Callback class for Trainer."""

    def __init__(self):
        """Initialize the callback."""

    def update_config(self, config: dict, *args, **kwargs):
        """Update the config of the callback."""

    def on_phase_start(self, phase: str):
        """Called when the phase starts."""

    def on_phase_end(self, phase: str):
        """Called when the phase ends."""

    def on_epoch_start(self, epoch: int, phase: str):
        """Called when the epoch starts."""

    def on_epoch_end(self, epoch: int, phase: str):
        """Called when the epoch ends."""

    def on_batch_start(self, step: int, phase: str):
        """Called when the batch starts."""

    def on_batch_end(self, step: int, phase: str, data: dict):
        """Called when the batch ends.

        Args:
            step: Current step number.
            phase: Current phase (train/val/test).
            data: Unified data container with namespaced keys
                (``input/*``, ``target``, ``pred``, ``post_processed/*``).
        """

    def on_lr_update(self, epoch: int, lr: float, scheduler_type: str = ""):
        """Called when the learning rate is updated."""

    def log_metrics(self, step: int, phase: str, metrics: dict[str, float]):
        """Log the metrics."""


class CallbackList:
    """list of callbacks."""

    def __init__(self, callbacks: list[Callback]):
        """Initialize the callback list."""
        self.callbacks = callbacks

    def __getattr__(self, attribute):
        """Get the attribute of the callback list."""

        def callback_wrapper(*args, **kwargs):
            for callback in self.callbacks:
                if hasattr(callback, attribute):
                    method = getattr(callback, attribute)
                    if callable(method):
                        method(*args, **kwargs)

        return callback_wrapper
