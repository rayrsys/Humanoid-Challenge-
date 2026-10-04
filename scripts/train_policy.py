"""Train the compact task-conditioned policy on retargeted human rollouts.

    python scripts/train_policy.py --demos outputs/demos.pkl --encoding factorised --out checkpoints/fact.pt
    python scripts/train_policy.py --demos outputs/demos.pkl --human-only --out checkpoints/human_only.pt
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from chessbot.policy import TaskPolicy, build_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="outputs/demos.pkl")
    ap.add_argument("--encoding", choices=["factorised", "flat"], default="factorised")
    ap.add_argument("--human-only", action="store_true", help="no symmetry augmentation")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with open(args.demos, "rb") as f:
        recs = pickle.load(f)
    recs = [r for r in recs if r["success"] and r["split"] == "train"]
    if args.human_only:
        recs = [r for r in recs if tuple(r["transform"]) == (0, 0, False)]
    tok, pose, delta, tgt = build_samples(recs, args.horizon)
    print(f"{len(recs)} episodes, {len(tok)} samples")
    tok, pose, delta, tgt = map(torch.from_numpy, (tok, pose, delta, tgt))

    model = TaskPolicy(args.encoding, args.horizon)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tok) // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(tok))
        total = 0.0
        for i in range(0, len(tok), args.batch):
            b = perm[i: i + args.batch]
            # Small state noise makes the closed loop robust to its own drift.
            noisy = pose[b] + 0.02 * torch.randn_like(pose[b])
            loss = (model(tok[b], noisy, delta[b]) - tgt[b]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}  L1 {total / len(tok):.4f}  ({time.time() - t0:.0f}s)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), encoding=args.encoding, horizon=args.horizon,
                    n_episodes=len(recs), human_only=args.human_only), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
