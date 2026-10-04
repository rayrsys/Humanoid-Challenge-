"""Chessboard geometry: square names <-> metric coordinates, and piece shapes.

Board frame (used for all actions and retargeting):
  * origin at the centre of the board, on its top surface
  * +x points from White's side towards Black's side (rank 1 -> rank 8)
  * +y points towards the a-file (White's left), so the frame is right-handed
  * +z points up

The robot sits on White's side looking down +x, so for the default layout the
board frame is the world frame translated by ``BoardGeometry.origin``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

FILES = "abcdefgh"


def parse_square(sq: str | int) -> int:
    """Accept ``"e4"`` or a python-chess square index."""
    if isinstance(sq, (int, np.integer)):
        return int(sq)
    return chess.parse_square(sq.lower())


@dataclass(frozen=True)
class PieceShape:
    """Primitive geometry for one piece type, in metres for a 5 cm square.

    ``parts`` are (geom type, size, z-centre, x-offset) tuples; ``grasp_z`` is
    the height above the board at which a parallel gripper should close.
    """

    height: float
    grasp_z: float
    grasp_radius: float
    parts: tuple


# Every piece has a straight "grip band" of radius grasp_radius around its
# grasp height. The Panda hand decides that height (see HAND_* below):
#   * finger pads span ~[pinch - 8 mm, pinch + 9 mm]; nothing wider than the
#     band may sit there, or the fingers pinch a rim and the piece pops out;
#   * the palm is 37 mm above the pinch point, so every piece is held within
#     33 mm of its top, otherwise the palm presses it into the board.
HAND_PALM_Z = 0.037       # palm underside above the pinch point
HAND_HALF_LENGTH = 0.104  # palm half-extent along the finger (closing) axis
HAND_HALF_WIDTH = 0.032   # palm half-extent across the finger axis
_BASE = (("cylinder", (0.016, 0.004), 0.004, 0.0), ("cylinder", (0.012, 0.002), 0.010, 0.0))


def _band(radius: float, top: float) -> tuple:
    return (("cylinder", (radius, (top - 0.012) / 2), (top + 0.012) / 2, 0.0),)


PIECE_SHAPES: dict[str, PieceShape] = {
    "p": PieceShape(0.0505, 0.023, 0.009, _BASE + _band(0.009, 0.034) + (
        ("sphere", (0.0095,), 0.041, 0.0),
    )),
    "r": PieceShape(0.050, 0.023, 0.011, _BASE + _band(0.011, 0.040) + (
        ("cylinder", (0.0125, 0.005), 0.045, 0.0),
    )),
    "n": PieceShape(0.056, 0.025, 0.010, _BASE + _band(0.010, 0.036) + (
        ("box", (0.010, 0.0065, 0.010), 0.046, 0.0),
    )),
    "b": PieceShape(0.064, 0.031, 0.009, _BASE + _band(0.009, 0.040) + (
        ("ellipsoid", (0.010, 0.010, 0.011), 0.049, 0.0),
        ("sphere", (0.0025,), 0.0615, 0.0),
    )),
    "q": PieceShape(0.064, 0.032, 0.010, _BASE + _band(0.010, 0.045) + (
        ("cylinder", (0.0125, 0.005), 0.050, 0.0),
        ("sphere", (0.0045,), 0.0595, 0.0),
    )),
    "k": PieceShape(0.068, 0.035, 0.0105, _BASE + _band(0.0105, 0.048) + (
        ("cylinder", (0.012, 0.004), 0.052, 0.0),
        ("box", (0.002, 0.002, 0.006), 0.062, 0.0),
        ("box", (0.006, 0.002, 0.002), 0.063, 0.0),
    )),
}

PIECE_NAMES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}


@dataclass(frozen=True)
class BoardGeometry:
    square: float = 0.05
    origin: tuple = (0.5, 0.0, 0.012)  # board-frame origin expressed in world
    thickness: float = 0.012
    border: float = 0.02
    _origin_arr: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_origin_arr", np.asarray(self.origin, dtype=float))

    @property
    def scale(self) -> float:
        """Piece scale relative to the 5 cm reference square."""
        return self.square / 0.05

    def square_xy(self, sq: str | int) -> np.ndarray:
        """Centre of ``sq`` in the board frame (metres, z omitted)."""
        s = parse_square(sq)
        f, r = chess.square_file(s), chess.square_rank(s)
        return np.array([(r - 3.5) * self.square, (3.5 - f) * self.square])

    def square_world(self, sq: str | int, z: float = 0.0) -> np.ndarray:
        xy = self.square_xy(sq)
        return self.to_world(np.array([xy[0], xy[1], z]))

    def xy_to_square(self, xy) -> int | None:
        """Square containing board-frame point ``xy``, or None if off the board."""
        r = int(np.floor(xy[0] / self.square + 4.0))
        f = int(np.floor(4.0 - xy[1] / self.square))
        if 0 <= r < 8 and 0 <= f < 8:
            return chess.square(f, r)
        return None

    def to_world(self, p_board) -> np.ndarray:
        return np.asarray(p_board, dtype=float) + self._origin_arr[: np.shape(p_board)[-1]]

    def to_board(self, p_world) -> np.ndarray:
        return np.asarray(p_world, dtype=float) - self._origin_arr[: np.shape(p_world)[-1]]

    def piece_shape(self, symbol: str) -> PieceShape:
        return PIECE_SHAPES[symbol.lower()]
