"""Meta-test: every src/ module should have a corresponding test file.

Auto-discovers all Python files under src/ and checks that tests/src/
mirrors them. Reports all missing test files in one failure message.
"""

from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_SRC = _ROOT / "src"
_TESTS_SRC = Path(__file__).parent / "src"

# Files that are intentionally untested (infrastructure, CLI tools, etc.)
_EXEMPT = {
    "__init__.py",
    "data_processor/compute_open_x_stats.py",
    "data_processor/download.py",
    "data_processor/downloaders/url_downloader.py",
    "data_processor/downloaders/kaggle_downloader.py",
    "data_processor/post_download_hooks/reorganizer.py",
    "dataset/dataset_server.py",
    "utils/distributed.py",
    "utils/wandb_utils.py",
    "utils/s3_cache_backend.py",
    "utils/remote_backup/base.py",
    "utils/remote_backup/ssh_strategy.py",
    "utils/remote_backup/wandb_strategy.py",
    "tools/checkpoint_surgeon.py",
}


def _discover_src_modules() -> list[str]:
    """Return all .py files under src/ as relative paths."""
    return sorted(str(p.relative_to(_SRC)) for p in _SRC.rglob("*.py") if p.name != "__init__.py")


def _expected_test_path(module_path: str) -> Path:
    """Map src/foo/bar.py -> tests/src/foo/test_bar.py."""
    parts = Path(module_path).parts
    parent = Path(*parts[:-1]) if len(parts) > 1 else Path()
    filename = f"test_{parts[-1]}"
    return _TESTS_SRC / parent / filename


@pytest.mark.unit
def test_every_src_module_has_test_file():
    """Every src/ Python module should have a test file under tests/src/."""
    modules = _discover_src_modules()
    missing = []
    for module in modules:
        rel = str(Path(module))
        if rel in _EXEMPT:
            continue
        test_path = _expected_test_path(module)
        if not test_path.exists():
            missing.append(f"  {module} -> {test_path.relative_to(_TESTS_SRC.parent)}")

    if missing:
        msg = (
            f"{len(missing)} src/ modules have no test file:\n"
            + "\n".join(missing)
            + "\n\nCreate test files or add to _EXEMPT if intentionally untested."
        )
        pytest.fail(msg)
