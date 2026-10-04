"""Render the kept Panda rollouts into a LeRobot v3 dataset (for SmolVLA etc.).

    MUJOCO_GL=egl python scripts/export_lerobot.py --demos artifacts/train_rollouts.jsonl.gz \\
        --repo-id <hf-user>/panda-chess-from-quest --root outputs/lerobot_panda_chess [--push]

Rollouts are replayed deterministically from their stored actions with the
front (agent-view) and wrist cameras on; simulation + rendering run in
parallel workers, the dataset is written in order by the main process.
Features:
    observation.images.front / .wrist  (256x256 video)
    observation.state  [x, y, z, yaw, width, q1..q7]   board frame, metres/rad
    action             [x, y, z, yaw, width]           executed Cartesian command
    task               the original teleop instruction, re-anchored for augmented copies
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no install needed

import argparse
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np

FPS = 20
SETTLE = 10
_W = {}


def _init(size):
    from chessbot.env import ChessEnv
    _W["env"] = ChessEnv(cameras=("front", "wrist"), image_size=(size, size))


def _render(r):
    import chess
    env = _W["env"]
    env.reset(fen=r["fen"])
    front, wrist, state, action = [], [], [], []
    for a in list(r["actions"]) + [r["actions"][-1]] * SETTLE:
        obs = env.observe()
        front.append(obs["images"]["front"])
        wrist.append(obs["images"]["wrist"])
        state.append(np.concatenate([obs["ee"], obs["qpos"]]).astype(np.float32))
        env.step(np.asarray(a))
        action.append(env.cmd.astype(np.float32))
    # Replays are deterministic, but stored actions are rounded: only keep
    # episodes that succeed again here, so no failure enters the dataset.
    ok = env.check_move(chess.Move.from_uci(r["src"] + r["dst"])).success
    return r["task"], np.stack(front), np.stack(wrist), np.stack(state), np.stack(action), bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="artifacts/train_rollouts.jsonl.gz")
    ap.add_argument("--repo-id", default="local/panda-chess-from-quest")
    ap.add_argument("--root", default="outputs/lerobot_panda_chess")
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--human-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--vcodec", default="libsvtav1")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from chessbot.rollouts import load_records

    recs = [r for r in load_records(args.demos) if r["success"] and r["split"] in args.splits]
    if args.human_only:
        recs = [r for r in recs if tuple(r["transform"]) == (0, 0, False)]
    if args.limit:
        recs = recs[: args.limit]
    print(f"exporting {len(recs)} episodes with {args.workers} render workers", flush=True)

    s = args.size
    features = {
        "observation.images.front": {"dtype": "video", "shape": (s, s, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "video", "shape": (s, s, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (12,),
                              "names": ["x", "y", "z", "yaw", "width"] + [f"q{i}" for i in range(1, 8)]},
        "action": {"dtype": "float32", "shape": (5,), "names": ["x", "y", "z", "yaw", "width"]},
    }
    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(args.repo_id, fps=FPS, features=features, root=root, robot_type="panda",
                               use_videos=True, vcodec=args.vcodec, image_writer_threads=4)
    dropped, written, t0 = 0, 0, time.time()
    chunk = 2 * args.workers  # bounded memory: render a chunk, write it, repeat
    with Pool(args.workers, initializer=_init, initargs=(s,)) as pool:
        for i in range(0, len(recs), chunk):
            for task, front, wrist, state, action, ok in pool.map(_render, recs[i: i + chunk]):
                if not ok:
                    dropped += 1
                    continue
                for t in range(len(front)):
                    ds.add_frame({"observation.images.front": front[t], "observation.images.wrist": wrist[t],
                                  "observation.state": state[t], "action": action[t], "task": task})
                ds.save_episode()
                written += 1
            el = time.time() - t0
            done = min(i + chunk, len(recs))
            print(f"  {done}/{len(recs)} episodes  ({el / 60:.1f} min, ~{el / done * (len(recs) - done) / 60:.0f} min left)",
                  flush=True)
    ds.finalize()
    print(f"wrote {root}: {written} episodes ({dropped} dropped because their replay did not succeed)")
    if args.push:
        ds.push_to_hub()


if __name__ == "__main__":
    main()
