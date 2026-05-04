"""Shim: run train.tools.optimize_hyperparams as __main__."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if __name__ == "__main__":
    runpy.run_module("train.tools.optimize_hyperparams", run_name="__main__")
