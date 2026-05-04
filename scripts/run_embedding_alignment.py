#!/usr/bin/env python3
"""
Demo: load and align multimodal embeddings to 1 fps.
Verifies embedding_aligner + optionally dumps alignment features.

Usage:
  python scripts/run_embedding_alignment.py [video_id] [--embeddings-root embeddings]
"""

import argparse
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Direct import to avoid loading heavy deps (whisper) from utils.__init__
from src.utils.embedding_aligner import get_alignment_features, load_aligned_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", nargs="?", default="1J5zlq2Vs3Y")
    parser.add_argument("--embeddings-root", default="embeddings")
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--alignment-only", action="store_true", help="Only compute alignment scalars (T,3)")
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    emb_root = os.path.join(root, args.embeddings_root)

    if args.alignment_only:
        aln = get_alignment_features(args.video_id, emb_root, args.duration)
        if aln is None:
            print("Missing embeddings, cannot compute alignment")
            return 1
        print(f"Alignment features shape: {aln.shape} (T, 3) = [vis_aud, vis_txt, aud_txt] cosine")
        print(f"Sample [0:5]:\n{aln[:5]}")
        return 0

    aligned, dur = load_aligned_embeddings(args.video_id, embeddings_root=emb_root, duration_sec=args.duration)
    print(f"Aligned embeddings: shape={aligned.shape}, duration={dur}s")
    print(f"  Dimensionality: visual 768 + audio 512 + text 256 = {aligned.shape[1]}")
    if aligned.size > 0:
        print(f"  Sample row 0: mean={aligned[0].mean():.4f}, std={aligned[0].std():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
