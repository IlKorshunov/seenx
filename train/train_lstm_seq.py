"""Shim: re-exports implementation from train.lstm.train_lstm_seq."""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from train.lstm.train_lstm_seq import main


if __name__ == "__main__":
    main()
