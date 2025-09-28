# Torch Module Node for the graph-based node system.

from typing import Any, Callable, Dict, Optional, Set

from torch import nn

from nodes.base import BaseNode


class TorchModuleNode(BaseNode, nn.Module):
    """Node that wraps a torch.nn.Module for graph-based execution."""

    def __init__(
        self,
        name: str,
        module: nn.Module,
        variable_name: str,
        run_modes: Optional[Set[str]] = None,
    ):
        """Initialize the TorchModuleNode.

        Args:
            name: Unique name for the node
            module: torch.nn.Module instance to wrap
            variable_name: Name of the variable to create for this module
            run_modes: Set of run modes where this node should execute
        """
        # Initialize BaseNode first
        BaseNode.__init__(
            self,
            name=name,
            run_modes=run_modes,
        )

        # Initialize nn.Module
        nn.Module.__init__(self)

        self._node_type = "torch_module"
        self._module = module
        self._variable_name = variable_name

    @property
    def module(self) -> nn.Module:
        """Get the wrapped module."""
        return self._module

    @property
    def variable_name(self) -> str:
        """Get the variable name."""
        return self._variable_name

    def __call__(self, *args, **kwargs):
        """Forward method for nn.Module compatibility."""
        return self._module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward method for nn.Module compatibility."""
        return self._module.forward(*args, **kwargs)

    def train(self, mode: bool = True):
        """Set training mode for both this node and wrapped module."""
        super().train(mode)
        self._module.train(mode)
        return self

    def eval(self):
        """Set evaluation mode for both this node and wrapped module."""
        super().eval()
        self._module.eval()
        return self

    def parameters(self, recurse: bool = True):
        """Return parameters of the wrapped module."""
        return self._module.parameters(recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        """Return named parameters of the wrapped module."""
        return self._module.named_parameters(prefix, recurse, remove_duplicate)

    def buffers(self, recurse: bool = True):
        """Return buffers of the wrapped module."""
        return self._module.buffers(recurse)

    def named_buffers(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """Return named buffers of the wrapped module."""
        return self._module.named_buffers(prefix, recurse, remove_duplicate)

    def children(self):
        """Return child modules of the wrapped module."""
        return self._module.children()

    def named_children(self):
        """Return named child modules of the wrapped module."""
        return self._module.named_children()

    def modules(self):
        """Return all modules of the wrapped module."""
        return self._module.modules()

    def named_modules(
        self, memo: Optional[Dict[str, Any]] = None, prefix: str = "", remove_duplicate: bool = True
    ):
        """Return named modules of the wrapped module."""
        return self._module.named_modules(memo, prefix, remove_duplicate)

    def apply(self, fn: Callable[[nn.Module], None]):
        """Apply function to all modules in the wrapped module."""
        self._module.apply(fn)
        return self

    def cuda(self, device=None):
        """Move wrapped module to CUDA."""
        self._module.cuda(device)
        return self

    def cpu(self):
        """Move wrapped module to CPU."""
        self._module.cpu()
        return self

    def to(self, *args, **kwargs):
        """Move wrapped module to device/dtype."""
        self._module.to(*args, **kwargs)
        return self

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients of wrapped module."""
        self._module.zero_grad(set_to_none)

    def share_memory(self):
        """Share memory of wrapped module."""
        self._module.share_memory()
        return self

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"wrapped_module={self._module.__class__.__name__}"

    def __getattr__(self, name):
        """Delegate attribute access to wrapped module if not found in this class."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate to wrapped module
            return getattr(self._module, name)

    def freeze(self) -> None:
        """Freeze the module by setting requires_grad=False for all parameters."""
        for param in self._module.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unfreeze the module by setting requires_grad=True for all parameters."""
        for param in self._module.parameters():
            param.requires_grad_(True)

    def __setattr__(self, name, value):
        """Set attribute, delegating to wrapped module for module-specific attributes."""
        # List of attributes that should be set on this object, not the wrapped module
        node_attrs = {
            "_name",
            "_node_type",
            "_run_modes",
            "_initialized",
            "_module",
            "_variable_name",
            "training",
        }

        if name in node_attrs or name.startswith("_") and hasattr(BaseNode, name):
            super().__setattr__(name, value)
        elif hasattr(self, "_module") and hasattr(self._module, name):
            # Delegate to wrapped module
            setattr(self._module, name, value)
        else:
            super().__setattr__(name, value)
