"""Read/write rollout records (pickle for local use, gzipped JSON lines to commit)."""

from __future__ import annotations

import gzip
import json
import pickle

import numpy as np

SLIM_KEYS = ("fen", "task", "piece", "colour", "src", "dst", "arm", "source_episode", "split",
             "transform", "success", "placement_error")


def load_records(path: str) -> list[dict]:
    if path.endswith(".jsonl.gz"):
        with gzip.open(path, "rt") as f:
            recs = [json.loads(line) for line in f]
        for r in recs:
            r["actions"] = np.asarray(r["actions"], np.float32)
            r["transform"] = tuple(r["transform"])
        return recs
    with open(path, "rb") as f:
        return pickle.load(f)


def save_slim(recs: list[dict], path: str):
    """Only what is needed to replay a rollout deterministically (actions + start state)."""
    with gzip.open(path, "wt") as f:
        for r in recs:
            d = {k: r[k] for k in SLIM_KEYS}
            d["transform"] = [int(r["transform"][0]), int(r["transform"][1]), bool(r["transform"][2])]
            d["actions"] = np.round(np.asarray(r["actions"], np.float64), 5).tolist()
            f.write(json.dumps(d) + "\n")
