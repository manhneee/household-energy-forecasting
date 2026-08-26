"""Build a synthetic processed panel when REFIT is not on disk yet."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.demo import write_demo_panel

if __name__ == "__main__":
    write_demo_panel()
