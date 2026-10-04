"""Play arbitrary chess moves with the human's own motions.

For a move (piece, src -> dst) we retrieve a teleop demo with the same piece
type and move vector (allowing the file mirror), re-anchor it onto the game's
squares with a board-symmetry transform, and execute it with the
object-centric retargeter. Pawns are absent from the teleop data, so they
borrow the motion of any piece with the same move vector; if no demo has the
exact vector, the closest one is used and its transport is warped to the
destination by the retargeter's keyframe pinning.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

import chess
import numpy as np

from .augment import transform
from .board import PIECE_SHAPES
from .oracle import YAW_CANDIDATES, palm_clearance
from .retarget import PIECE_SYMBOL, HumanTrajectory, retarget_object_centric

PIECE_NAME = {v: k for k, v in PIECE_SYMBOL.items()}


def _vec(src: str, dst: str) -> tuple[int, int]:
    s, d = chess.parse_square(src), chess.parse_square(dst)
    return chess.square_file(d) - chess.square_file(s), chess.square_rank(d) - chess.square_rank(s)


class HumanMotionPlayer:
    def __init__(self, trajectories: list[HumanTrajectory], seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.index = defaultdict(list)  # (piece, df, dr) -> [(ht, mirror)]
        for ht in trajectories:
            e = ht.episode
            df, dr = _vec(e.src, e.dst)
            self.index[(e.piece, df, dr)].append((ht, False))
            self.index[(e.piece, -df, dr)].append((ht, True))
        self.used = []

    def retrieve(self, piece: str, src: str, dst: str):
        df, dr = _vec(src, dst)
        exact = self.index.get((piece, df, dr)) or [c for (p, a, b), v in self.index.items()
                                                     if (a, b) == (df, dr) for c in v]
        if exact:
            return exact[self.rng.integers(len(exact))]
        keys = sorted(self.index, key=lambda k: ((k[1] - df) ** 2 + (k[2] - dr) ** 2, k[0] != piece))
        cands = self.index[keys[0]]
        return cands[self.rng.integers(len(cands))]

    def plan(self, env, a: int, b: int) -> np.ndarray:
        src, dst = chess.square_name(a), chess.square_name(b)
        name = env.occupant[a]
        colour, sym = ("white" if name[0] == "w" else "black"), name.split("_")[1]
        piece = PIECE_NAME[sym]
        ht, mirror = self.retrieve(piece, src, dst)
        e = ht.episode
        s0 = chess.parse_square(e.src)
        f0 = 7 - chess.square_file(s0) if mirror else chess.square_file(s0)
        moved = transform(ht, chess.square_file(a) - f0, chess.square_rank(a) - chess.square_rank(s0), mirror)
        # Re-label for the game: the actual piece, colour and destination.
        ep = dataclasses.replace(moved.episode, piece=piece, colour=colour, dst=dst, fen=env.board.fen(),
                                 task=f"Move the {colour} {piece} from {src} to {dst}.")
        moved = dataclasses.replace(moved, episode=ep)
        self.used.append((src + dst, e.episode_id, mirror))

        k = env.geom.scale
        obstacles = [(env.geom.square_xy(sq), PIECE_SHAPES[n.split("_")[1]].height * k)
                     for sq, n in env.occupant.items() if sq != a]
        gz = PIECE_SHAPES[sym].grasp_z * k

        def safe_yaw(yaw_g, _yaw_r):
            """Keep the human's yaw unless the palm would clip a neighbour."""
            ignore = (a, b)
            def gap(y):
                return min(palm_clearance(env, env.geom.square_xy(a), gz, y, ignore),
                           palm_clearance(env, env.geom.square_xy(b), gz, y, ignore))
            if gap(yaw_g) >= 0.004:
                return yaw_g, yaw_g
            best = max(YAW_CANDIDATES, key=lambda y: (gap(y) >= 0.004, -abs(np.angle(np.exp(2j * (y - yaw_g))))))
            return best, best

        return retarget_object_centric(moved, env.geom, trim=True, obstacles=obstacles, yaw_fn=safe_yaw,
                                       start=env.cmd)

    def mover(self, env, a: int, b: int, on_step=None):
        """``game.execute_move``-compatible mover."""
        for act in self.plan(env, a, b):
            env.step(act)
            if on_step:
                on_step(env)
        for _ in range(8):
            env.step(env.cmd)
            if on_step:
                on_step(env)
