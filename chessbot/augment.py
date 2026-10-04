"""Multiply human demonstrations using the chessboard's symmetries.

A demo moving a piece from ``src`` to ``dst`` is re-anchored to every
translated copy (src + v, dst + v) that stays on the board, optionally after
mirroring files (a <-> h). Both operations preserve the move's legality for
its piece type (a knight's L stays an L), and because the human trajectory
lives in continuous board coordinates, the whole path - approach, transport
style, timing, wrist heading - moves rigidly with it. Every candidate is then
replayed on the Panda and only physically successful ones are kept.
"""

from __future__ import annotations

import dataclasses

import chess
import numpy as np

from .retarget import HumanTrajectory


def _shift(sq: str, df: int, dr: int, mirror: bool) -> str | None:
    s = chess.parse_square(sq)
    f, r = chess.square_file(s), chess.square_rank(s)
    if mirror:
        f = 7 - f
    f, r = f + df, r + dr
    if 0 <= f < 8 and 0 <= r < 8:
        return chess.square_name(chess.square(f, r))
    return None


def _transform_fen(fen: str, src: str, dst: str, df: int, dr: int, mirror: bool, rng) -> str:
    """Move every piece rigidly; anything pushed off the board (or onto the
    destination) is re-dropped on a random free square away from the move."""
    board = chess.Board(fen)
    new = chess.Board(None)
    new.turn = board.turn
    moving = board.piece_at(chess.parse_square(src))
    new_src, new_dst = _shift(src, df, dr, mirror), _shift(dst, df, dr, mirror)
    new.set_piece_at(chess.parse_square(new_src), moving)
    keep_clear = {chess.parse_square(new_src), chess.parse_square(new_dst)}
    homeless = []
    for sq, piece in board.piece_map().items():
        if sq == chess.parse_square(src):
            continue
        t = _shift(chess.square_name(sq), df, dr, mirror)
        if t is None or chess.parse_square(t) in keep_clear or new.piece_at(chess.parse_square(t)):
            homeless.append(piece)
        else:
            new.set_piece_at(chess.parse_square(t), piece)
    for piece in homeless:
        free = [s for s in chess.SQUARES if new.piece_at(s) is None and
                min(chess.square_distance(s, k) for k in keep_clear) >= 2]
        new.set_piece_at(free[rng.integers(len(free))], piece)
    return new.fen()


def transform(ht: HumanTrajectory, df: int, dr: int, mirror: bool = False, seed: int = 0) -> HumanTrajectory | None:
    e = ht.episode
    src, dst = _shift(e.src, df, dr, mirror), _shift(e.dst, df, dr, mirror)
    if src is None or dst is None:
        return None
    fr = ht.fr.copy()
    heading = ht.heading.copy()
    if mirror:
        fr[:, 0] = 7.0 - fr[:, 0]
        heading = np.pi - heading
    fr += np.array([df, dr], float)
    rng = np.random.default_rng(seed)
    fen = _transform_fen(e.fen, e.src, e.dst, df, dr, mirror, rng)
    task = f"Move the {e.colour} {e.piece} from {src} to {dst}."
    ep = dataclasses.replace(e, src=src, dst=dst, fen=fen, task=task,
                             episode_id=f"{e.episode_id}_t{df:+d}{dr:+d}{'m' if mirror else ''}")
    return dataclasses.replace(ht, episode=ep, fr=fr, heading=heading)


def all_transforms(ht: HumanTrajectory, mirror: bool = True, seed: int = 0):
    """Every on-board translation (and mirrored translation) of a demo, excluding identity."""
    out = []
    for m in ([False, True] if mirror else [False]):
        for df in range(-7, 8):
            for dr in range(-7, 8):
                if df == 0 and dr == 0 and not m:
                    continue
                t = transform(ht, df, dr, m, seed=seed + 1000 * df + 31 * dr + int(m))
                if t is not None:
                    out.append(t)
    return out
