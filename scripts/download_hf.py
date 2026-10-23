"""Download model weights from Hugging Face into $PC_ROOT/ckpt (or --out).

Usage (needs your own HF_TOKEN in the environment; SAM 3 is gated and access is already granted):
    export HF_TOKEN=...
    python scripts/download_hf.py --repo facebook/sam3 --out $PC_ROOT/ckpt/sam3
    python scripts/download_hf.py --repo IDEA-Research/grounding-dino-base --out $PC_ROOT/ckpt/grounding-dino-base

Verify the repo ids on Hugging Face if a download fails.
"""

import argparse
import os

from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. facebook/sam3")
    ap.add_argument("--out", required=True, help="local directory to store the files")
    args = ap.parse_args()
    path = snapshot_download(repo_id=args.repo, local_dir=args.out, token=os.environ.get("HF_TOKEN"))
    print("downloaded to", path)


if __name__ == "__main__":
    main()
