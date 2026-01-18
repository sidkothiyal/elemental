"""
Pytest configuration and fixtures for config tests.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def config_dir():
    """Return the path to the configs directory."""
    return Path(__file__).parent.parent / "configs"


@pytest.fixture(scope="session")
def flat_configs(config_dir):
    """Return list of all flat config names."""
    configs = []
    for f in config_dir.glob("*.yaml"):
        name = f.stem
        if name not in ["config", "default"] and not name.startswith("node_train"):
            configs.append(name)
    return configs
