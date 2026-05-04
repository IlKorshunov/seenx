"""Shim: re-exports implementation from train.tools.compare_fixed_inference_models."""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from train.tools.compare_fixed_inference_models import main


if __name__ == "__main__":
    main()
