"""Download the Quest 3 teleop datasets from the Hugging Face Hub into ./data.

    python scripts/download_data.py            # parquet + metadata only (~50 MB)
    python scripts/download_data.py --videos   # also the head/wrist camera videos (~1.7 GB)
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

DATASETS = ["Raysolo/fb32-v03-all5-right", "Raysolo/fb32-v03-all5-left"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    patterns = ["data/*", "meta/*", "provenance.json", "README.md"] + (["videos/*"] if args.videos else [])
    for repo in DATASETS:
        path = snapshot_download(repo, repo_type="dataset", local_dir=f"{args.out}/{repo.split('/')[1]}",
                                 allow_patterns=patterns)
        print("downloaded", repo, "->", path)


if __name__ == "__main__":
    main()
