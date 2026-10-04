"""Compact task-conditioned policy trained on retargeted human demonstrations.

The instruction ("Move the white knight from h1 to f2.") is parsed into
tokens - piece, colour, source square, destination square - and squares are
encoded either

* ``factorised``: separate file and rank embeddings, summed, so a square never
  seen in training ("f5") is still composed from a seen file and a seen rank;
* ``flat``: one embedding per square (no way to generalise to unseen squares).

The network sees the end-effector state and its last change, and predicts a
chunk of future Cartesian commands (ACT-style chunking, executed receding
horizon). It is deliberately small: it trains on a laptop CPU in minutes.
"""

from __future__ import annotations

import re

import chess
import numpy as np
import torch
from torch import nn

PIECES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
COLOURS = ["white", "black"]
TASK_RE = re.compile(r"Move the (white|black) (\w+) from ([a-h][1-8]) to ([a-h][1-8])")

# Normalisation for the board-frame action/state [x, y, z, yaw, width].
POS_SCALE = np.array([0.2, 0.2, 0.1], np.float32)
WIDTH_SCALE = 0.08


def parse_task(task: str) -> tuple[str, str, str, str]:
    m = TASK_RE.search(task)
    if not m:
        raise ValueError(f"unparseable task: {task!r}")
    return m.group(1), m.group(2), m.group(3), m.group(4)


def task_tokens(task: str) -> np.ndarray:
    colour, piece, src, dst = parse_task(task)
    s, d = chess.parse_square(src), chess.parse_square(dst)
    return np.array([PIECES.index(piece), COLOURS.index(colour),
                     chess.square_file(s), chess.square_rank(s),
                     chess.square_file(d), chess.square_rank(d)], np.int64)


def encode_pose(p: np.ndarray) -> np.ndarray:
    """[x, y, z, yaw, width] -> normalised [x, y, z, sin2yaw, cos2yaw, width] (gripper is pi-symmetric)."""
    p = np.asarray(p, np.float32)
    return np.concatenate([p[..., :3] / POS_SCALE, np.sin(2 * p[..., 3:4]), np.cos(2 * p[..., 3:4]),
                           p[..., 4:5] / WIDTH_SCALE], -1)


def decode_pose(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    yaw = 0.5 * np.arctan2(v[..., 3], v[..., 4])
    return np.concatenate([v[..., :3] * POS_SCALE, yaw[..., None], v[..., 5:6] * WIDTH_SCALE], -1)


class TaskPolicy(nn.Module):
    def __init__(self, encoding: str = "factorised", horizon: int = 10, width: int = 512, emb: int = 64):
        super().__init__()
        self.encoding, self.horizon = encoding, horizon
        self.piece = nn.Embedding(len(PIECES), emb)
        self.colour = nn.Embedding(len(COLOURS), emb)
        if encoding == "factorised":
            self.file = nn.ModuleList([nn.Embedding(8, emb) for _ in range(2)])
            self.rank = nn.ModuleList([nn.Embedding(8, emb) for _ in range(2)])
        else:
            self.square = nn.ModuleList([nn.Embedding(64, emb) for _ in range(2)])
        d_in = 4 * emb + 12  # piece, colour, src, dst + pose(6) + delta(6)
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, horizon * 6),
        )

    def embed_task(self, tok: torch.Tensor) -> torch.Tensor:
        parts = [self.piece(tok[:, 0]), self.colour(tok[:, 1])]
        for i in range(2):
            f, r = tok[:, 2 + 2 * i], tok[:, 3 + 2 * i]
            if self.encoding == "factorised":
                parts.append(self.file[i](f) + self.rank[i](r))
            else:
                parts.append(self.square[i](r * 8 + f))
        return torch.cat(parts, -1)

    def forward(self, tok, pose, delta):
        x = torch.cat([self.embed_task(tok), pose, delta], -1)
        return self.net(x).view(-1, self.horizon, 6)


def build_samples(records: list[dict], horizon: int):
    """(tokens, pose, delta, target_chunk) arrays from successful rollouts."""
    toks, poses, deltas, targets = [], [], [], []
    for r in records:
        ee = encode_pose(r["ee"])
        cmd = encode_pose(r["cmd"])
        T = len(cmd)
        tok = task_tokens(r["task"])
        idx = np.arange(T)
        chunk = np.minimum(idx[:, None] + np.arange(horizon)[None], T - 1)
        toks.append(np.repeat(tok[None], T, 0))
        poses.append(ee[:T])
        deltas.append(np.concatenate([np.zeros_like(ee[:1]), np.diff(ee[:T], axis=0)]) * 10.0)
        targets.append(cmd[chunk])
    return (np.concatenate(toks), np.concatenate(poses).astype(np.float32),
            np.concatenate(deltas).astype(np.float32), np.concatenate(targets).astype(np.float32))


class PolicyRunner:
    """Closed-loop execution in ChessEnv with receding-horizon chunks."""

    def __init__(self, model: TaskPolicy, execute: int = 5):
        self.model = model.eval()
        self.execute = execute

    @torch.no_grad()
    def run(self, env, task: str, max_steps: int = 300, on_step=None) -> int:
        tok = torch.from_numpy(task_tokens(task))[None]
        prev = env.ee_state()
        steps = 0
        while steps < max_steps:
            cur = env.ee_state()
            pose = torch.from_numpy(encode_pose(cur))[None]
            delta = torch.from_numpy((encode_pose(cur) - encode_pose(prev)) * 10.0)[None]
            chunk = decode_pose(self.model(tok, pose, delta)[0].numpy())
            for a in chunk[: self.execute]:
                prev = env.ee_state()
                env.step(a)
                steps += 1
                if on_step:
                    on_step(env)
                if steps >= max_steps:
                    break
        return steps
