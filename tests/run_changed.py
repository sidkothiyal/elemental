#!/usr/bin/env python3
"""Run pytest only for test files that correspond to changed source files.

Usage
-----
    uv run python tests/run_changed.py              # compare HEAD vs main
    uv run python tests/run_changed.py HEAD~1        # compare HEAD vs previous commit
    uv run python tests/run_changed.py origin/main   # compare HEAD vs remote main

The script maps each changed file to its test counterpart using the
following rules:

* ``src/<path>.py``         → ``tests/src/<dir>/test_<file>.py``
* ``configs/<anything>``    → ``tests/integration/test_config_loading.py``
* ``tests/<path>``          → that test file itself (run directly)

Any mapped test file that does not exist on disk is silently skipped.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def get_changed_files(base: str) -> list[str]:
    """Return a list of files changed between base and HEAD."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def map_to_tests(changed: list[str]) -> set[str]:
    """Map changed source files to their corresponding test files."""
    tests: set[str] = set()
    for f in changed:
        path = Path(f)
        if f.startswith("src/") and f.endswith(".py") and path.name != "__init__.py":
            # src/model/unet.py → tests/src/model/test_unet.py
            rel = path.relative_to("src")
            test_path = ROOT / "tests" / "src" / rel.parent / f"test_{rel.name}"
            if test_path.exists():
                tests.add(str(test_path))
        elif f.startswith("configs/"):
            tests.add(str(ROOT / "tests" / "integration" / "test_config_loading.py"))
        elif f.startswith("tests/") and f.endswith(".py"):
            full = ROOT / f
            if full.exists():
                tests.add(str(full))
    return tests


def main():
    """Entry point: discover changed files, map to tests, and run pytest."""
    base = sys.argv[1] if len(sys.argv) > 1 else "main"
    changed = get_changed_files(base)
    if not changed or changed == [""]:
        print("No changed files detected.")
        sys.exit(0)

    print(f"Changed files vs {base!r}:")
    for f in changed:
        print(f"  {f}")

    tests = map_to_tests(changed)
    if not tests:
        print("\nNo matching test files found for the changed files.")
        sys.exit(0)

    print(f"\nRunning {len(tests)} test file(s):")
    for t in sorted(tests):
        print(f"  {t}")

    cmd = ["uv", "run", "pytest"] + sorted(tests) + ["-v", "--tb=short"]
    result = subprocess.run(cmd, check=False, cwd=ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
