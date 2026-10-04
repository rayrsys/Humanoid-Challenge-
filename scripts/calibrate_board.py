"""Recover the teleop board pose from the demonstrations and save it to calib/.

    python scripts/calibrate_board.py            # both arms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np

from chessbot.calibration import calibrate
from chessbot.quest_data import load_dataset

DATASETS = {"right": "fb32-v03-all5-right", "left": "fb32-v03-all5-left"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder containing the datasets (default: ./data)")
    ap.add_argument("--out", default="calib")
    args = ap.parse_args()
    Path(args.out).mkdir(exist_ok=True)
    cals = {}
    for arm, name in DATASETS.items():
        eps = load_dataset(name, args.data)
        cal, diag = calibrate(eps, iters=4)
        cal.save(Path(args.out) / f"board_{arm}.json")
        cals[arm] = cal
        res = diag["residual"] * 1000
        kinds = np.array([r[5] for r in diag["rows"]])
        print(f"[{arm}] {len(eps)} episodes, square {cal.square * 1000:.2f} mm, anisotropy {cal.anisotropy:.3f}, "
              f"grasp-point offset {np.round(cal.offset, 3)} m")
        print(f"       residual: grasp median {np.median(res[kinds == 'grasp']):.1f} mm, "
              f"release median {np.median(res[kinds == 'release']):.1f} mm, rms {cal.rms_xy_mm:.1f} mm "
              f"({cal.n_events} events after trimming)")
        print(f"       refinement (rms, p95, n): {[tuple(round(v, 1) for v in h) for h in diag['history']]}")
    d = [np.linalg.norm(cals["right"].square_xy(chess.square_name(q)) - cals["left"].square_xy(chess.square_name(q)))
         for q in range(64)]
    print(f"left vs right board estimate: mean {np.mean(d) * 1000:.1f} mm, max {np.max(d) * 1000:.1f} mm apart")


if __name__ == "__main__":
    main()
