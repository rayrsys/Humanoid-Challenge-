"""Render the kept Panda rollouts into a LeRobot v3 dataset (for SmolVLA etc.).

    MUJOCO_GL=egl python scripts/export_lerobot.py --demos outputs/demos.pkl \\
        --repo-id <hf-user>/panda-chess-from-quest --root outputs/lerobot_panda_chess [--push]

Rollouts are replayed deterministically from their stored actions with the
front (agent-view) and wrist cameras on. Features:
    observation.images.front / .wrist  (256x256 video)
    observation.state  [x, y, z, yaw, width, q1..q7]   board frame, metres/rad
    action             [x, y, z, yaw, width]           executed Cartesian command
    task               the original teleop instruction, re-anchored for augmented copies
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

FPS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="outputs/demos.pkl")
    ap.add_argument("--repo-id", default="local/panda-chess-from-quest")
    ap.add_argument("--root", default="outputs/lerobot_panda_chess")
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--human-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--vcodec", default="libsvtav1")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    import chess
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from chessbot.env import ChessEnv
    from chessbot.rollouts import load_records

    recs = [r for r in load_records(args.demos) if r["success"] and r["split"] in args.splits]
    if args.human_only:
        recs = [r for r in recs if tuple(r["transform"]) == (0, 0, False)]
    if args.limit:
        recs = recs[: args.limit]
    print(f"exporting {len(recs)} episodes")

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
    env = ChessEnv(cameras=("front", "wrist"), image_size=(s, s))
    mismatches = 0
    for i, r in enumerate(recs):
        env.reset(fen=r["fen"])
        for a in r["actions"].tolist() + [r["actions"][-1].tolist()] * 10:
            obs = env.observe()
            state = np.concatenate([obs["ee"], obs["qpos"]]).astype(np.float32)
            env.step(np.asarray(a))
            ds.add_frame({
                "observation.images.front": obs["images"]["front"],
                "observation.images.wrist": obs["images"]["wrist"],
                "observation.state": state,
                "action": env.cmd.astype(np.float32),
                "task": r["task"],
            })
        # Replays are deterministic, but stored actions are rounded: only keep
        # episodes that succeed again here, so no failure enters the dataset.
        if not env.check_move(chess.Move.from_uci(r["src"] + r["dst"])).success:
            mismatches += 1
            ds.clear_episode_buffer()
            continue
        ds.save_episode()
        if i % 50 == 0:
            print(f"  {i}/{len(recs)}", flush=True)
    ds.finalize()
    print(f"wrote {root} ({mismatches} episodes dropped because their replay did not succeed)")
    if args.push:
        ds.push_to_hub()


if __name__ == "__main__":
    main()
