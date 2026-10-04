"""Open-loop replay of retargeted teleop demos on the Panda chess env.

    MUJOCO_GL=egl python scripts/replay.py --mode object_centric --arm both
    MUJOCO_GL=egl python scripts/replay.py --mode naive --limit 40 --videos 4

Writes per-episode results to outputs/replay_<mode>.json and prints success
rates per arm and piece type.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no install needed

import argparse
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

DATASETS = {"right": "fb32-v03-all5-right", "left": "fb32-v03-all5-left"}
_W = {}


def _init(mode, speed, data, record):
    from chessbot.calibration import BoardCalibration
    from chessbot.env import ChessEnv
    from chessbot.quest_data import load_dataset

    _W["env"] = ChessEnv(cameras=())
    _W["cal"] = {a: BoardCalibration.load(f"calib/board_{a}.json") for a in DATASETS}
    _W["eps"] = {a: {e.index: e for e in load_dataset(n, data)} for a, n in DATASETS.items()}
    _W["mode"], _W["speed"], _W["record"] = mode, speed, record


def obstacles(env, src):
    """(xy, height) of every standing piece except the one being moved."""
    import chess
    from chessbot.board import PIECE_SHAPES

    out = []
    for sq, name in env.occupant.items():
        if sq != chess.parse_square(src):
            out.append((env.geom.square_xy(sq), PIECE_SHAPES[name.split("_")[1]].height * env.geom.scale))
    return out


def run_one(job):
    import chess
    from chessbot.retarget import extract, retarget_naive, retarget_object_centric

    arm, idx = job
    env, e = _W["env"], _W["eps"][arm][idx]
    ht = extract(e, _W["cal"][arm])
    env.reset(fen=e.fen)
    if _W["mode"] == "naive":
        actions = retarget_naive(ht, env.geom, _W["speed"])
    else:
        actions = retarget_object_centric(ht, env.geom, _W["speed"], obstacles=obstacles(env, e.src))
    frames = []
    for a in actions:
        env.step(a)
        if (arm, idx) in _W["record"] and env.t % 2 == 0:
            frames.append(env.render("player", size=(360, 480)))
    for _ in range(10):  # let the piece settle
        env.step(actions[-1])
    r = env.check_move(chess.Move.from_uci(e.src + e.dst))
    if frames:
        import imageio
        imageio.mimsave(f"outputs/replay_{_W['mode']}_{arm}_{idx}.mp4", frames, fps=10)
    return dict(arm=arm, episode=idx, piece=e.piece, colour=e.colour, task=e.task, src=e.src, dst=e.dst,
                steps=len(actions), **{k: (v if not isinstance(v, np.generic) else v.item())
                                       for k, v in r.as_dict().items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["naive", "object_centric"], default="object_centric")
    ap.add_argument("--arm", choices=["right", "left", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="episodes per arm (0 = all)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--videos", type=int, default=0, help="record the first N episodes per arm")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    from chessbot.quest_data import load_dataset

    arms = list(DATASETS) if args.arm == "both" else [args.arm]
    jobs = []
    for a in arms:
        idxs = [e.index for e in load_dataset(DATASETS[a], args.data) if e.labelled]
        jobs += [(a, i) for i in (idxs[: args.limit] if args.limit else idxs)]
    record = {j for a in arms for j in [x for x in jobs if x[0] == a][: args.videos]}
    t0 = time.time()
    from chessbot.assets import panda_dir
    panda_dir()  # fetch the robot model once, before workers start
    with Pool(args.workers, initializer=_init, initargs=(args.mode, args.speed, args.data, record)) as pool:
        results = pool.map(run_one, jobs, chunksize=4)
    Path("outputs").mkdir(exist_ok=True)
    out = Path(f"outputs/replay_{args.mode}.json")
    out.write_text(json.dumps(results, indent=1))

    by = defaultdict(list)
    for r in results:
        by[(r["arm"], r["piece"])].append(r["success"])
        by[(r["arm"], "ALL")].append(r["success"])
        by[("ALL", "ALL")].append(r["success"])
    print(f"mode={args.mode} speed={args.speed}  ({len(results)} episodes, {time.time() - t0:.0f}s)")
    for k in sorted(by):
        print(f"  {k[0]:5s} {k[1]:7s} success {np.mean(by[k]):6.1%}  (n={len(by[k])})")
    fails = defaultdict(int)
    for r in results:
        if not r["success"]:
            why = ("not released" if not r["released"] else "tipped over" if not r["upright"]
                   else "knocked other piece" if r["disturbed"] else
                   f"wrong square ({r['placed_square']})" if r["placed_square"] != r["dst"] else "off-centre")
            fails[why.split(" (")[0]] += 1
    print("  failure modes:", dict(fails))
    print("wrote", out)


if __name__ == "__main__":
    main()
