"""Callback class for Trainer."""
from typing import List


class Callback:
    """Callback class for Trainer."""

    def __init__(self):
        """Initialize the callback."""

    def update_config(self, *args, **kwargs):
        """Update the config of the callback."""

    def on_train_start(self):
        """Called when the training starts."""

    def on_train_end(self):
        """Called when the training ends."""

    def on_epoch_start(self):
        """Called when the epoch starts."""

    def on_epoch_end(self):
        """Called when the epoch ends."""

    def on_batch_start(self):
        """Called when the batch starts."""

    def on_batch_end(self):
        """Called when the batch ends."""

    def on_validation_start(self):
        """Called when the validation starts."""

    def on_validation_end(self):
        """Called when the validation ends."""


class CallbackList:
    """List of callbacks."""

    # TODO: Make this cleaner instead of redefining all the methods

    def __init__(self, callbacks: List[Callback]):
        """Initialize the callback list."""
        self.callbacks = callbacks

    def update_config(self, *args, **kwargs):
        """Update the config of the callback list."""
        for callback in self.callbacks:
            callback.update_config(*args, **kwargs)

    def on_train_start(self):
        """Called when the training starts."""
        for callback in self.callbacks:
            callback.on_train_start()

    def on_train_end(self):
        """Called when the training ends."""
        for callback in self.callbacks:
            callback.on_train_end()

    def on_epoch_start(self):
        """Called when the epoch starts."""
        for callback in self.callbacks:
            callback.on_epoch_start()

    def on_epoch_end(self):
        """Called when the epoch ends."""
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_batch_start(self):
        """Called when the batch starts."""
        for callback in self.callbacks:
            callback.on_batch_start()

    def on_batch_end(self):
        """Called when the batch ends."""
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_validation_start(self):
        """Called when the validation starts."""
        for callback in self.callbacks:
            callback.on_validation_start()

    def on_validation_end(self):
        """Called when the validation ends."""
        for callback in self.callbacks:
            callback.on_validation_end()
