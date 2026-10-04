"""Closed-loop evaluation on seen tasks, unseen square pairs and unseen squares.

    python scripts/eval_policy.py --ckpt checkpoints/fact.pt checkpoints/flat.pt --n 60
    python scripts/eval_policy.py --reference   # replay the retargeted human motion itself

Evaluation tasks are symmetry transforms of the human demos (so every task is
a legal move with a matching human motion), sampled with a fixed seed per
split; every model sees exactly the same tasks and starting positions.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DATASETS = {"right": "fb32-v03-all5-right", "left": "fb32-v03-all5-left"}
SPLITS = ("train", "unseen_pair", "unseen_square")
_W = {}


def eval_tasks(n: int, seed: int = 123, data=None):
    """{split: [(arm, episode_index, df, dr, mirror)]} with n tasks per split."""
    import chess
    from chessbot.quest_data import load_dataset
    from chessbot.splits import split_of

    rng = np.random.default_rng(seed)
    pools = defaultdict(list)
    for arm, name in DATASETS.items():
        for e in load_dataset(name, data):
            if not e.labelled:
                continue
            s, d = chess.parse_square(e.src), chess.parse_square(e.dst)
            for m in (False, True):
                fs, fd = (7 - chess.square_file(s), 7 - chess.square_file(d)) if m else \
                    (chess.square_file(s), chess.square_file(d))
                for df in range(-7, 8):
                    for dr in range(-7, 8):
                        a, b = (fs + df, chess.square_rank(s) + dr), (fd + df, chess.square_rank(d) + dr)
                        if all(0 <= v < 8 for v in (*a, *b)):
                            sp = split_of(e.piece, chess.square_name(chess.square(*a)),
                                          chess.square_name(chess.square(*b)))
                            pools[sp].append((arm, e.index, df, dr, m))
    return {sp: [pools[sp][i] for i in rng.choice(len(pools[sp]), min(n, len(pools[sp])), replace=False)]
            for sp in SPLITS}


def _init(ckpt, data):
    import torch
    from chessbot.calibration import BoardCalibration
    from chessbot.env import ChessEnv
    from chessbot.policy import PolicyRunner, TaskPolicy
    from chessbot.quest_data import load_dataset
    from chessbot.retarget import extract

    torch.set_num_threads(1)
    _W["env"] = ChessEnv(cameras=())
    _W["ht"] = {}
    for arm, name in DATASETS.items():
        cal = BoardCalibration.load(f"calib/board_{arm}.json")
        for e in load_dataset(name, data):
            if e.labelled:
                _W["ht"][(arm, e.index)] = extract(e, cal)
    if ckpt:
        c = torch.load(ckpt, weights_only=False)
        model = TaskPolicy(c["encoding"], c["horizon"])
        model.load_state_dict(c["state"])
        _W["runner"] = PolicyRunner(model)


def _run(job):
    import chess
    from chessbot.augment import transform
    from scripts.generate_demos import rollout

    split, (arm, idx, df, dr, m) = job
    ht = _W["ht"][(arm, idx)]
    if (df, dr, m) != (0, 0, False):
        ht = transform(ht, df, dr, m, seed=hash((arm, idx, df, dr, m)) % 2**31)
    e = ht.episode
    env = _W["env"]
    if "runner" in _W:
        env.reset(fen=e.fen)
        _W["runner"].run(env, e.task)
        r = env.check_move(chess.Move.from_uci(e.src + e.dst))
        ok, err = r.success, r.placement_error
    else:
        rec, ok = rollout(env, ht)
        err = rec["placement_error"]
    return dict(split=split, task=e.task, piece=e.piece, success=bool(ok), placement_error=float(err))


def evaluate(ckpt, tasks, workers, data):
    jobs = [(sp, t) for sp in SPLITS for t in tasks[sp]]
    with Pool(workers, initializer=_init, initargs=(ckpt, data)) as pool:
        return pool.map(_run, jobs, chunksize=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--reference", action="store_true", help="also evaluate retargeted human replay")
    ap.add_argument("--n", type=int, default=60, help="tasks per split")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="outputs/eval.json")
    args = ap.parse_args()

    tasks = eval_tasks(args.n, data=args.data)
    runs = ([("human replay (reference)", None)] if args.reference else []) + [(Path(c).stem, c) for c in args.ckpt]
    table = {}
    for name, ckpt in runs:
        res = evaluate(ckpt, tasks, args.workers, args.data)
        table[name] = {sp: float(np.mean([r["success"] for r in res if r["split"] == sp])) for sp in SPLITS}
        table[name]["per_piece"] = {p: float(np.mean([r["success"] for r in res if r["piece"] == p]))
                                    for p in sorted({r["piece"] for r in res})}
        print(f"{name:28s} " + "  ".join(f"{sp} {table[name][sp]:6.1%}" for sp in SPLITS), flush=True)
    prev = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    prev.update(table)
    Path(args.out).write_text(json.dumps(prev, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
