"""Compatibility wrapper for running tests.

The project uses pytest-based tests under `tests/`.
`python tests.py` or `main.py --test` (if wired) may call `run_all_tests()`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_all_tests() -> bool:
    """Run pytest and return True/False for compatibility."""
    try:
        import pytest

        code = int(pytest.main(["tests", "-q"]))
        return code == 0
    except Exception as e:
        logger.error("pytest run failed: %s", e)
        return False


if __name__ == "__main__":
    ok = run_all_tests()
    raise SystemExit(0 if ok else 1)
