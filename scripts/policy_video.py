"""Record the trained policy executing instructions (captioned, with a wrist-camera inset).

    MUJOCO_GL=egl python scripts/policy_video.py --ckpt checkpoints/aug_factorised.pt \\
        --per-split 1 1 2 --out outputs/policy_in_action.mp4
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SPLIT_LABEL = {"train": "seen task", "unseen_pair": "unseen square pair", "unseen_square": "UNSEEN square"}


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def caption(frame, wrist, lines, banner=None):
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, img.width, 50), fill=(20, 20, 20))
    d.text((10, 6), lines[0], fill=(255, 255, 255), font=_font(17))
    d.text((10, 29), lines[1], fill=(200, 200, 200), font=_font(13))
    w = Image.fromarray(wrist).resize((150, 150))
    img.paste(w, (img.width - 158, img.height - 158))
    d.rectangle((img.width - 159, img.height - 159, img.width - 7, img.height - 7), outline=(255, 255, 255), width=2)
    if banner:
        ok = banner == "success"
        d.rectangle((10, img.height - 46, 150, img.height - 12), fill=(27, 140, 80) if ok else (200, 60, 50))
        d.text((22, img.height - 40), "SUCCESS" if ok else "FAILED", fill=(255, 255, 255), font=_font(18))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/aug_factorised.pt")
    ap.add_argument("--per-split", type=int, nargs=3, default=[1, 1, 2], help="clips for train/unseen_pair/unseen_square")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs/policy_in_action.mp4")
    ap.add_argument("--camera", default="front")
    args = ap.parse_args()

    import chess
    import imageio
    import torch

    from chessbot.augment import transform
    from chessbot.calibration import BoardCalibration
    from chessbot.env import ChessEnv
    from chessbot.policy import PolicyRunner, TaskPolicy
    from chessbot.quest_data import load_dataset
    from chessbot.retarget import extract
    from scripts.eval_policy import DATASETS, SPLITS, eval_tasks

    c = torch.load(args.ckpt, weights_only=False)
    model = TaskPolicy(c["encoding"], c["horizon"])
    model.load_state_dict(c["state"])
    runner = PolicyRunner(model)
    hts = {}
    for arm, name in DATASETS.items():
        cal = BoardCalibration.load(f"calib/board_{arm}.json")
        for e in load_dataset(name):
            if e.labelled:
                hts[(arm, e.index)] = extract(e, cal)

    env = ChessEnv(cameras=())
    tasks = eval_tasks(20, seed=args.seed)
    frames = []
    for sp, k in zip(SPLITS, args.per_split):
        for arm, idx, df, dr, m in tasks[sp][:k]:
            ht = hts[(arm, idx)]
            if (df, dr, m) != (0, 0, False):
                ht = transform(ht, df, dr, m, seed=hash((arm, idx, df, dr, m)) % 2**31)
            e = ht.episode
            env.reset(fen=e.fen)
            lines = [f'"{e.task}"', f"policy input: this instruction + gripper state  |  {SPLIT_LABEL[sp]}"]
            clip = []

            def rec(env_):
                clip.append(caption(env_.render(args.camera, size=(360, 480)),
                                    env_.render("wrist", size=(150, 150)), lines))
            runner.run(env, e.task, on_step=rec)
            ok = env.check_move(chess.Move.from_uci(e.src + e.dst)).success
            last = caption(env.render(args.camera, size=(360, 480)), env.render("wrist", size=(150, 150)), lines,
                           banner="success" if ok else "failed")
            frames += clip + [last] * 25
            print(f"[{sp}] {e.task}: {'success' if ok else 'failed'}", flush=True)
    imageio.mimsave(args.out, frames, fps=20, macro_block_size=8)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
