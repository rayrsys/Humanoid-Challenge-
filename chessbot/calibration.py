"""Recover the teleop chessboard's pose from the demonstrations alone.

The datasets record the G1 wrist pose but not where the board was. Every
episode, however, says which squares were touched (source, destination) and
the hand closure tells us *when*. Assuming the board is level and the grasp
point sits at a fixed offset ``o`` in the wrist frame, at every grasp/release

    p_wrist[:2] + (R_wrist @ o)[:2] = origin + A @ [file, rank]

which is *linear* in (origin, A, o): 9 unknowns, two equations per event,
~1350 events. ``A`` comes out as a 2x2 matrix; its singular values are the
square size and their ratio checks that the fitted board is actually square.
Heights are fitted the same way with one grasp height per piece type.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import chess
import numpy as np

from .quest_data import Episode, grasp_events


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """(N, 4) w-x-y-z quaternions -> (N, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


@dataclass
class BoardCalibration:
    origin: list          # (x, y) of the a1 centre in the robot frame
    A: list               # 2x2: robot xy = origin + A @ [file, rank]
    z_board: float        # board surface height (from the lowest grasp heights)
    offset: list          # grasp point in the wrist frame (m)
    piece_grasp_z: dict   # mean grasp-point height above z_board per piece type
    square: float         # mean square size (m)
    anisotropy: float     # ratio of A's singular values (1 = square board)
    rms_xy_mm: float
    p95_xy_mm: float
    n_events: int
    quat_order: str

    def square_xy(self, sq: str) -> np.ndarray:
        s = chess.parse_square(sq)
        return np.asarray(self.origin) + np.asarray(self.A) @ np.array([chess.square_file(s), chess.square_rank(s)])

    def to_board(self, p: np.ndarray) -> np.ndarray:
        """Robot-frame points -> continuous board coords [file, rank, height/square].

        File/rank are in square units with a1's centre at (0, 0); height is
        above the board surface, also in square units.
        """
        A_inv = np.linalg.inv(np.asarray(self.A))
        fr = (A_inv @ (p[..., :2] - np.asarray(self.origin)).T).T
        h = (p[..., 2] - self.z_board) / self.square
        return np.concatenate([fr, h[..., None]], axis=-1)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path) -> "BoardCalibration":
        with open(path) as f:
            return cls(**json.load(f))


def flexion_events(e: Episode, close: float = 0.9, open_: float = 0.6, min_hold: int = 15):
    """Grasp/release from the two finger-flexion joints (reliable for power grasps only)."""
    closed, start, best = False, None, None
    for i, g in enumerate(e.grip):
        if not closed and g > close:
            closed, start = True, i
        elif closed and g < open_:
            closed = False
            if i - start >= min_hold and (best is None or i - start > best[1] - best[0]):
                best = (start, i)
    return best


def grasp_point(e: Episode, offset) -> np.ndarray:
    """(T, 3) fingertip contact point: wrist position + wrist rotation @ offset."""
    return e.pos + np.einsum("tij,j->ti", quat_wxyz_to_mat(e.quat), np.asarray(offset))


def segment_by_labels(e: Episode, cal: "BoardCalibration", height_weight: float = 1.0,
                      min_gap: float = 0.3) -> tuple[int, int]:
    """Grasp/release indices = closest low approach to the labelled squares.

    Cost (in squares) = horizontal distance to the square + weighted height
    above the episode's lowest point; grasp is searched first, release after it.
    """
    b = cal.to_board(grasp_point(e, cal.offset))
    h = b[:, 2] - b[:, 2].min()
    src = np.array(cal_square(e.src))
    dst = np.array(cal_square(e.dst))
    cost_g = np.linalg.norm(b[:, :2] - src, axis=1) + height_weight * h
    cost_r = np.linalg.norm(b[:, :2] - dst, axis=1) + height_weight * h
    # Grasp must precede the closest approach to the destination.
    t_dst = int(np.argmin(cost_r))
    g = int(np.argmin(cost_g[: max(t_dst, 1)]))
    gap = int(min_gap * 50)
    r = g + gap + int(np.argmin(cost_r[g + gap:])) if g + gap < len(cost_r) else len(cost_r) - 1
    return g, r


def cal_square(sq: str) -> tuple[int, int]:
    s = chess.parse_square(sq)
    return chess.square_file(s), chess.square_rank(s)


def _events(episodes: list[Episode], segment=None):
    """(wrist_pos, wrist_quat, square, piece, episode, kind) at every grasp and release."""
    rows = []
    for e in episodes:
        if not e.labelled:
            continue
        ev = segment(e) if segment else grasp_events(e.hand)
        if ev is None:
            continue
        g, r = ev
        rows.append((e.pos[g], e.quat[g], e.src, e.piece, e.index, "grasp"))
        rows.append((e.pos[r], e.quat[r], e.dst, e.piece, e.index, "release"))
    return rows


def calibrate(episodes: list[Episode], iters: int = 3) -> tuple["BoardCalibration", dict]:
    """Bootstrap from clean power grasps, then re-segment everything by labels and refit."""
    power = [e for e in episodes if e.grip.max() > 1.0]
    cal, diag = fit_board(power, segment=flexion_events)
    history = [(cal.rms_xy_mm, cal.p95_xy_mm, cal.n_events)]
    for _ in range(iters):
        cal, diag = fit_board(episodes, segment=lambda e, c=cal: segment_by_labels(e, c))
        history.append((cal.rms_xy_mm, cal.p95_xy_mm, cal.n_events))
    diag["history"] = history
    return cal, diag


def fit_board(episodes: list[Episode], quat_order: str = "wxyz", trim: float = 3.0,
              segment=None) -> tuple[BoardCalibration, dict]:
    rows = _events(episodes, segment)
    P = np.array([r[0] for r in rows])
    Q = np.array([r[1] for r in rows])
    if quat_order == "xyzw":
        Q = Q[:, [3, 0, 1, 2]]
    R = quat_wxyz_to_mat(Q)
    FR = np.array([[chess.square_file(chess.parse_square(r[2])), chess.square_rank(chess.parse_square(r[2]))]
                   for r in rows], float)

    # Unknowns: [ox, oy, a11, a12, a21, a22, o1, o2, o3]
    n = len(rows)
    M = np.zeros((2 * n, 9))
    b = np.zeros(2 * n)
    for i in range(n):
        f, k = FR[i]
        M[2 * i, :6] = [1, 0, f, k, 0, 0]
        M[2 * i + 1, :6] = [0, 1, 0, 0, f, k]
        M[2 * i, 6:] = -R[i, 0]
        M[2 * i + 1, 6:] = -R[i, 1]
        b[2 * i: 2 * i + 2] = P[i, :2]
    keep = np.ones(n, bool)
    for _ in range(3):  # robust refit: drop events > trim * median residual
        rows_keep = np.repeat(keep, 2)
        x, *_ = np.linalg.lstsq(M[rows_keep], b[rows_keep], rcond=None)
        res = np.linalg.norm((M @ x - b).reshape(n, 2), axis=1)
        keep = res < trim * np.median(res) + 1e-4
    origin, A, o = x[:2], x[2:6].reshape(2, 2), x[6:]

    # Heights: p_z + (R o)_z = z_board + h_piece
    gz = P[:, 2] + np.einsum("nj,j->n", R[:, 2], o)
    pieces = sorted({r[3] for r in rows})
    per_piece = {p: float(np.median(gz[[r[3] == p for r in rows]])) for p in pieces}
    sv = np.linalg.svd(A, compute_uv=False)
    square = float(sv.mean())
    z_board = float(np.percentile(gz, 1)) - 0.01 * square / 0.05  # small margin below the lowest grasp
    cal = BoardCalibration(
        origin=origin.tolist(), A=A.tolist(), z_board=z_board, offset=o.tolist(),
        piece_grasp_z={p: v - z_board for p, v in per_piece.items()},
        square=square, anisotropy=float(sv[0] / sv[1]),
        rms_xy_mm=float(np.sqrt(np.mean(res[keep] ** 2)) * 1000),
        p95_xy_mm=float(np.percentile(res, 95) * 1000), n_events=int(keep.sum()), quat_order=quat_order,
    )
    diag = dict(residual=res, keep=keep, rows=rows, grasp_height=gz)
    return cal, diag
