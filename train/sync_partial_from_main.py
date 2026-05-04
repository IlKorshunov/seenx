"""Shim: re-exports implementation from train.tools.sync_partial_from_main."""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from train.tools.sync_partial_from_main import main


if __name__ == "__main__":
    main()
