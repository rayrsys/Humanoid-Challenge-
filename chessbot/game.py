"""Turn chess moves into physical pick-and-place primitives.

A *mover* is any callable ``mover(env, from_sq, to_sq) -> None`` that tries to
physically carry the piece on ``from_sq`` to ``to_sq`` (scripted oracle or a
learned policy). Chess rules around it are handled here:

  * captures: the captured piece is cleared to the graveyard first (teleported:
    we do not manipulate captures, and say so in the README)
  * castling: two primitives, king then rook
  * en passant: the passed pawn is cleared, then the capturing pawn moves
  * promotion: the pawn moves, then is swapped for a spare queen in place
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import mujoco
import numpy as np


@dataclass
class MoveOutcome:
    move: str
    primitives: list = field(default_factory=list)  # (uci, success, placement_error)

    @property
    def success(self) -> bool:
        return all(p[1] for p in self.primitives)


def _primitive(env, mover, a: int, b: int, out: MoveOutcome) -> bool:
    uci = chess.square_name(a) + chess.square_name(b)
    mover(env, a, b)
    res = env.check_move(chess.Move(a, b))
    out.primitives.append((uci, res.success, res.placement_error))
    # Keep the logical map in sync with where the piece physically is.
    env.occupant[b] = env.occupant.pop(a)
    return res.success


def execute_move(env, move: chess.Move, mover) -> MoveOutcome:
    board = env.board
    out = MoveOutcome(move.uci())

    if board.is_en_passant(move):
        env.remove_piece(move.to_square + (-8 if board.turn == chess.WHITE else 8))
    elif board.is_capture(move):
        env.remove_piece(move.to_square)

    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)
        _primitive(env, mover, move.from_square, move.to_square, out)
        _primitive(env, mover, rook_from, rook_to, out)
    else:
        _primitive(env, mover, move.from_square, move.to_square, out)

    if move.promotion:
        _promote(env, move.to_square, chess.Piece(move.promotion, board.turn))

    board.push(move)
    return out


def _promote(env, sq: int, piece: chess.Piece):
    """Swap the pawn on ``sq`` for a spare piece of the promoted type."""
    pawn = env.occupant.pop(sq)
    color = "w" if piece.color else "b"
    sym = piece.symbol().lower()
    used = set(env.occupant.values())
    spare = next(n for n in env.names if n.startswith(f"{color}_{sym}_") and n not in used)
    env._park(pawn)
    xy = env.geom.square_xy(sq)
    env._place(spare, xy)
    env.occupant[sq] = spare
    mujoco.mj_forward(env.model, env.data)


def random_game_moves(n: int, seed: int = 0, fen: str = chess.STARTING_FEN) -> list[chess.Move]:
    rng = np.random.default_rng(seed)
    board = chess.Board(fen)
    moves = []
    for _ in range(n):
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = legal[rng.integers(len(legal))]
        moves.append(mv)
        board.push(mv)
    return moves
