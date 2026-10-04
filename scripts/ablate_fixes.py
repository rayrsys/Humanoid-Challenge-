"""Leave-one-out ablation of the simulator, controller and retargeting fixes.

    MUJOCO_GL=egl python scripts/ablate_fixes.py [--workers 4]

Replays every labelled teleop demo with the object-centric retargeter (as
``replay.py`` does) once per variant, each time with one fix switched off, and
appends one JSON line per variant to outputs/ablation.jsonl.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no install needed

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np

# variant: (ChessEnv kwargs, retarget_object_centric kwargs)
VARIANTS = {
    "nothing removed": ({}, {}),
    "setpoint ramp": ({"ramp_setpoints": False}, {}),
    "speed cap": ({}, {"v_max": np.inf, "yaw_rate_max": np.inf}),
    "neighbour clearance": ({}, {"neighbour_clearance": False}),
    "start transit": ({}, {"start": None}),
    "IK warm start": ({"ik_warm_start": False}, {}),
    "finger coupling": ({"couple_fingers": False}, {}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="outputs/ablation.jsonl")
    args = ap.parse_args()

    from chessbot.assets import panda_dir
    from chessbot.quest_data import load_dataset
    from scripts.replay import DATASETS, _init, run_one

    panda_dir()  # fetch the robot model once, before workers start
    jobs = [(a, e.index) for a, n in DATASETS.items() for e in load_dataset(n, args.data) if e.labelled]
    Path(args.out).parent.mkdir(exist_ok=True)
    for name in args.variants:
        env_kw, rt_kw = VARIANTS[name]
        with Pool(args.workers, initializer=_init,
                  initargs=("object_centric", 1.0, args.data, set(), env_kw, rt_kw)) as pool:
            res = pool.map(run_one, jobs, chunksize=4)
        row = dict(removed=name, success=float(np.mean([r["success"] for r in res])), n=len(res),
                   failures=dict(Counter(r["failure"] for r in res if not r["success"])))
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
