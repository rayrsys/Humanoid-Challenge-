"""Closed-loop evaluation of a fine-tuned SmolVLA checkpoint in the Panda chess env.

    MUJOCO_GL=egl python scripts/eval_smolvla.py \\
        --ckpt outputs/train/smolvla_panda_chess/checkpoints/last/pretrained_model --n 30 --video 3

Uses the same task sets as ``eval_policy.py`` (seen tasks, unseen pairs,
unseen squares) and LeRobot's own pre/post-processing pipeline, mirroring
``lerobot_eval``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no install needed

import argparse
import json

import numpy as np

RENAME = {"observation.images.front": "observation.images.camera1",
          "observation.images.wrist": "observation.images.camera2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=30, help="tasks per split")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--continuous-gripper", dest="binary_gripper", action="store_false",
                    help="execute predicted gripper widths as is (default: snap to closed/open like the demos)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--video", type=int, default=0, help="save videos of the first N tasks per split")
    ap.add_argument("--out", default="outputs/eval_smolvla.json")
    args = ap.parse_args()

    import chess
    import imageio
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from chessbot.augment import transform
    from chessbot.calibration import BoardCalibration
    from chessbot.env import ChessEnv
    from chessbot.policy import binarise_gripper
    from chessbot.quest_data import load_dataset
    from chessbot.retarget import extract
    from scripts.eval_policy import DATASETS, SPLITS, eval_tasks

    policy = SmolVLAPolicy.from_pretrained(args.ckpt).to(args.device).eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device},
                                "rename_observations_processor": {"rename_map": RENAME}})

    hts = {}
    for arm, name in DATASETS.items():
        cal = BoardCalibration.load(f"calib/board_{arm}.json")
        for e in load_dataset(name):
            if e.labelled:
                hts[(arm, e.index)] = extract(e, cal)

    env = ChessEnv(cameras=("front", "wrist"), image_size=(args.size, args.size))
    tasks = eval_tasks(args.n)
    results = {}
    for sp in SPLITS:
        ok = []
        for k, (arm, idx, df, dr, m) in enumerate(tasks[sp]):
            ht = hts[(arm, idx)]
            if (df, dr, m) != (0, 0, False):
                ht = transform(ht, df, dr, m, seed=hash((arm, idx, df, dr, m)) % 2**31)
            e = ht.episode
            obs = env.reset(fen=e.fen)
            policy.reset()
            frames = []
            for _ in range(args.max_steps):
                batch = {
                    "observation.images.front": torch.from_numpy(obs["images"]["front"]).permute(2, 0, 1)[None].float() / 255,
                    "observation.images.wrist": torch.from_numpy(obs["images"]["wrist"]).permute(2, 0, 1)[None].float() / 255,
                    "observation.state": torch.from_numpy(np.concatenate([obs["ee"], obs["qpos"]]).astype(np.float32))[None],
                    "task": [e.task],
                }
                with torch.inference_mode():
                    action = post(policy.select_action(pre(batch)))
                a = action[0].detach().cpu().numpy()
                obs = env.step(binarise_gripper(a) if args.binary_gripper else a)
                if k < args.video:
                    frames.append(env.render("player", size=(360, 480)))
            r = env.check_move(chess.Move.from_uci(e.src + e.dst))
            ok.append(r.success)
            if frames:
                imageio.mimsave(f"outputs/smolvla_{sp}_{k}.mp4", frames, fps=20)
            print(f"[{sp}] {e.task}: {'OK' if r.success else 'fail'} ({r.placement_error * 1000:.0f} mm)", flush=True)
        results[sp] = float(np.mean(ok))
    print(json.dumps(results, indent=2))
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
