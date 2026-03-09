"""Pytest configuration for performance benchmarks."""

import os
import sys
from pathlib import Path

# Ensure project root is importable when running benchmarks from performance/.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Keep numba cache in workspace-local directory for reproducibility.
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / ".numba_cache"
_DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = str(_DEFAULT_CACHE_DIR)
os.environ["NUMBA_DISABLE_FILE_SYSTEM_CACHE"] = "1"


def pytest_configure(config) -> None:
    """Register benchmark markers for this test tree."""
    config.addinivalue_line("markers", "slow: marks benchmarks as slow")
