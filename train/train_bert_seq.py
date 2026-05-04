"""Shim: re-exports implementation from train.bert.train_bert_seq."""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from train.bert.train_bert_seq import main


if __name__ == "__main__":
    main()
