"""Stress-test the simulator: the scripted oracle plays random legal games.

Reports per-primitive success so we know the env is solvable before any
learning happens. Optionally records a video of the first game.

    MUJOCO_GL=egl python scripts/oracle_selfplay.py --games 3 --moves 30 --video outputs/oracle_game.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no install needed

import argparse
import json
import time

import imageio
import numpy as np

from chessbot.env import ChessEnv
from chessbot.game import execute_move, random_game_moves
from chessbot.oracle import oracle_mover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--moves", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--camera", default="player")
    args = ap.parse_args()

    env = ChessEnv(cameras=())
    frames = []
    results = []
    t0 = time.time()
    for g in range(args.games):
        env.reset(seed=args.seed + g)
        record = args.video and g == 0

        def mover(env, a, b):
            def cb(_obs):
                if record and env.t % 2 == 0:
                    frames.append(env.render(args.camera, size=(360, 480)))
            oracle_mover(env, a, b, on_step=cb)

        for mv in random_game_moves(args.moves, seed=args.seed + g):
            out = execute_move(env, mv, mover)
            results.extend(out.primitives)
            if not out.success:
                print(f"game {g} move {out.move}: FAILED {out.primitives}")
    ok = np.array([r[1] for r in results])
    err = np.array([r[2] for r in results]) * 1000
    summary = dict(primitives=len(ok), success_rate=float(ok.mean()),
                   placement_err_mm_mean=float(err.mean()), placement_err_mm_p95=float(np.percentile(err, 95)),
                   seconds=round(time.time() - t0, 1))
    print(json.dumps(summary, indent=2))
    if args.video and frames:
        imageio.mimsave(args.video, frames, fps=20)
        print("wrote", args.video)


if __name__ == "__main__":
    main()
