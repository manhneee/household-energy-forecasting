"""Build the hourly panel, weather cache, and data_quality.md."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pipeline import build_hourly_panel


if __name__ == "__main__":
    build_hourly_panel()
