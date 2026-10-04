"""Retarget + augment the teleop demos and keep only those that succeed on the Panda.

    MUJOCO_GL=egl python scripts/generate_demos.py --aug-per-demo 4 --out outputs/demos.pkl

Each record stores the starting FEN, task string, the executed action
sequence (the rate-limited Cartesian command the controller tracked), the
end-effector states, and provenance (source episode, transform). Images are
not rendered here; ``export_lerobot.py`` replays the kept rollouts
deterministically with cameras on.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

DATASETS = {"right": "fb32-v03-all5-right", "left": "fb32-v03-all5-left"}
_W = {}


def _init(data):
    from chessbot.calibration import BoardCalibration
    from chessbot.env import ChessEnv
    from chessbot.quest_data import load_dataset
    from chessbot.retarget import extract

    _W["env"] = ChessEnv(cameras=())
    _W["ht"] = {}
    for arm, name in DATASETS.items():
        cal = BoardCalibration.load(f"calib/board_{arm}.json")
        for e in load_dataset(name, data):
            if e.labelled:
                _W["ht"][(arm, e.index)] = extract(e, cal)


def rollout(env, ht, speed=1.0):
    """Run one retargeted trajectory; returns (record, success)."""
    import chess
    from chessbot.retarget import retarget_object_centric
    from scripts.replay import obstacles

    e = ht.episode
    env.reset(fen=e.fen)
    actions = retarget_object_centric(ht, env.geom, speed, trim=True, obstacles=obstacles(env, e.src))
    states, cmds = [], []
    for a in actions:
        states.append(env.ee_state())
        env.step(a)
        cmds.append(env.cmd.copy())
    for _ in range(10):
        states.append(env.ee_state())
        env.step(actions[-1])
        cmds.append(env.cmd.copy())
    r = env.check_move(chess.Move.from_uci(e.src + e.dst))
    rec = dict(fen=e.fen, task=e.task, piece=e.piece, colour=e.colour, src=e.src, dst=e.dst,
               arm=e.arm, source_episode=e.episode_id, actions=np.asarray(actions, np.float32),
               cmd=np.asarray(cmds, np.float32), ee=np.asarray(states, np.float32),
               placement_error=float(r.placement_error))
    return rec, bool(r.success)


def work(job):
    from chessbot.augment import transform
    from chessbot.splits import split_of

    key, df, dr, mirror = job
    ht = _W["ht"][key]
    if (df, dr, mirror) != (0, 0, False):
        ht = transform(ht, df, dr, mirror, seed=hash(job) % 2**31)
        if ht is None:
            return None
    e = ht.episode
    rec, ok = rollout(_W["env"], ht)
    rec["split"] = split_of(e.piece, e.src, e.dst)
    rec["transform"] = (df, dr, mirror)
    rec["success"] = ok
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-per-demo", type=int, default=4)
    ap.add_argument("--out", default="outputs/demos.pkl")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import chess
    from chessbot.quest_data import load_dataset
    from chessbot.splits import split_of

    rng = np.random.default_rng(args.seed)
    jobs = []
    for arm, name in DATASETS.items():
        for e in load_dataset(name, args.data):
            if not e.labelled:
                continue
            key = (arm, e.index)
            jobs.append((key, 0, 0, False))  # the human demo itself
            # Candidate symmetries that keep the move on the board and in the train split.
            s, d = chess.parse_square(e.src), chess.parse_square(e.dst)
            cands = []
            for m in (False, True):
                fs, fd = (7 - chess.square_file(s), 7 - chess.square_file(d)) if m else \
                    (chess.square_file(s), chess.square_file(d))
                for df in range(-7, 8):
                    for dr in range(-7, 8):
                        if (df, dr, m) == (0, 0, False):
                            continue
                        a = (fs + df, chess.square_rank(s) + dr)
                        b = (fd + df, chess.square_rank(d) + dr)
                        if not all(0 <= v < 8 for v in (*a, *b)):
                            continue
                        src = chess.square_name(chess.square(*a))
                        dst = chess.square_name(chess.square(*b))
                        if split_of(e.piece, src, dst) == "train":
                            cands.append((df, dr, m))
            for i in rng.permutation(len(cands))[: args.aug_per_demo]:
                jobs.append((key, *cands[i]))
    print(f"{len(jobs)} rollouts to run")
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.data,)) as pool:
        recs = [r for r in pool.imap_unordered(work, jobs, chunksize=8) if r is not None]
    kept = [r for r in recs if r["success"]]
    human = [r for r in recs if r["transform"] == (0, 0, False)]
    print(f"done in {time.time() - t0:.0f}s: {len(kept)}/{len(recs)} successful "
          f"(human-only {np.mean([r['success'] for r in human]):.1%}, "
          f"augmented {np.mean([r['success'] for r in recs if r['transform'] != (0, 0, False)]):.1%})")
    print("kept by split:", Counter(r["split"] for r in kept))
    with open(args.out, "wb") as f:
        pickle.dump(recs, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
