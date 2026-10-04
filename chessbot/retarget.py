"""Retarget G1 + Dex3 teleop demonstrations onto the Panda chess env.

Both robots play on a chessboard, so the shared frame is the *board*: every
human trajectory is first expressed in continuous board coordinates (file,
rank in square units, height in metres) using the per-arm calibration, and
then re-embodied on the Panda's board.

Two retargeters, deliberately contrasting:

* ``naive``: geometric copy. Board-frame grasp-point path scaled to the
  Panda's board, gripper yaw from the wrist heading, gripper width from the
  normalised hand closure. This is what "just retarget the hand" gives you.

* ``object_centric``: keeps the human's path *shape* and *timing* but anchors
  it to the task: the grasp/release keyframes (found from the labels, see
  ``calibration.segment_by_labels``) are pinned to the piece's grasp point on
  the Panda's board, and the residual offset is blended in and out smoothly
  (MimicGen-style segment transforms). The gripper closes exactly between
  the keyframes, carries above the tallest piece, and yaw follows the human
  but avoids palm collisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from .board import PIECE_SHAPES, BoardGeometry
from .calibration import BoardCalibration, grasp_point, quat_wxyz_to_mat, segment_by_labels
from .env import CONTROL_HZ, MAX_WIDTH
from .quest_data import FPS, Episode, closure

PIECE_SYMBOL = {"pawn": "p", "knight": "n", "bishop": "b", "rook": "r", "queen": "q", "king": "k"}


@dataclass
class HumanTrajectory:
    """One demonstration in the board frame (time base: dataset FPS)."""
    episode: Episode
    fr: np.ndarray        # (T, 2) continuous [file, rank], a1 centre = (0, 0)
    height: np.ndarray    # (T,) grasp-point height above the board surface (m)
    heading: np.ndarray   # (T,) wrist pointing direction in the board plane (rad, file axis = 0)
    closure: np.ndarray   # (T,) normalised hand closure in [0, 1]
    t_grasp: int
    t_release: int
    square: float         # teleop square size (m)

    @property
    def piece_symbol(self) -> str:
        return PIECE_SYMBOL[self.episode.piece]


def extract(e: Episode, cal: BoardCalibration) -> HumanTrajectory:
    gp = grasp_point(e, cal.offset)
    b = cal.to_board(gp)
    # Wrist x-axis (towards the fingers) projected onto the board plane, in board axes.
    R = quat_wxyz_to_mat(e.quat)
    fwd = R[:, :2, 0]
    fwd_board = (np.linalg.inv(np.asarray(cal.A)) @ fwd.T).T
    heading = np.unwrap(np.arctan2(fwd_board[:, 1], fwd_board[:, 0]))
    c = closure(e.hand)
    g, r = segment_by_labels(e, cal)
    return HumanTrajectory(e, b[:, :2], b[:, 2] * cal.square, heading, c / max(c.max(), 1e-6), g, r, cal.square)


def _resample(x: np.ndarray, n_out: int) -> np.ndarray:
    t_in = np.linspace(0.0, 1.0, len(x))
    t_out = np.linspace(0.0, 1.0, n_out)
    if x.ndim == 1:
        return np.interp(t_out, t_in, x)
    return np.stack([np.interp(t_out, t_in, x[:, i]) for i in range(x.shape[1])], 1)


def _board_to_sim(fr: np.ndarray, geom: BoardGeometry) -> np.ndarray:
    """Continuous [file, rank] -> Panda board-frame xy (see board.py)."""
    return np.stack([(fr[:, 1] - 3.5) * geom.square, (3.5 - fr[:, 0]) * geom.square], 1)


def _heading_to_yaw(heading: np.ndarray) -> np.ndarray:
    """Board-frame heading (file axis = 0) -> Panda gripper yaw, folded to (-pi/2, pi/2].

    The Panda's fingers close perpendicular to the direction a human hand
    points when pinching from the side, which is the convention used here.
    In the sim frame the file axis is -y and the rank axis is +x.
    """
    ang = heading - np.pi / 2  # rotate board axes into sim axes (file -> -y)
    yaw = ang
    return (yaw + np.pi / 2) % np.pi - np.pi / 2


def _n_steps(T: int, speed: float) -> int:
    return max(2, int(round(T / FPS * CONTROL_HZ / speed)))


def retarget_naive(ht: HumanTrajectory, geom: BoardGeometry, speed: float = 1.0) -> np.ndarray:
    n = _n_steps(len(ht.fr), speed)
    xy = _board_to_sim(_resample(ht.fr, n), geom)
    z = _resample(ht.height, n) * geom.square / ht.square
    yaw = _heading_to_yaw(_resample(ht.heading, n))
    width = MAX_WIDTH * (1.0 - np.clip(_resample(ht.closure, n), 0, 1))
    return np.column_stack([xy, z, np.unwrap(yaw * 2) / 2, width])


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _polyline(points, fracs, n: int) -> np.ndarray:
    """n samples along a piecewise-linear path through ``points`` at time fractions ``fracs``."""
    t = np.linspace(0.0, 1.0, n)
    pts = np.asarray(points, float)
    return np.stack([np.interp(t, fracs, pts[:, i]) for i in range(pts.shape[1])], 1)


def retarget_object_centric(ht: HumanTrajectory, geom: BoardGeometry, speed: float = 1.0,
                            blend_s: float = 0.8, funnel_s: float = 0.6, lift_s: float = 0.4,
                            dwell_s: float = 0.4, arrive_s: float = 0.3, hover: float = 0.05,
                            carry_z: float = 0.12, trim: bool = False, lead_s: float = 1.0,
                            tail_s: float = 0.6, obstacles=(), yaw_fn=None, return_source_index: bool = False):
    """Human timing, path shape and yaw; Panda-appropriate contact funnels.

    Phases (dataset frames): approach | pre-grasp funnel | grasp dwell | lift |
    transport | pre-place funnel | release dwell | retreat | rest.
    ``obstacles`` are (xy, height) of other standing pieces in the board frame.
    ``trim`` drops the G1's rest pose: the sequence starts ``lead_s`` before
    the pre-grasp funnel and ends ``tail_s`` after the retreat (the Panda
    starts from its own home pose, so the G1's is meaningless here).
    """
    T = len(ht.fr)
    k = geom.scale
    shape = PIECE_SHAPES[ht.piece_symbol]
    gz = shape.grasp_z * k
    pre_w = 2 * shape.grasp_radius * k + 0.024
    e = ht.episode
    S = geom.square

    # 1. Human grasp-point path on the Panda board, pinned to the task keyframes.
    p = np.column_stack([_board_to_sim(ht.fr, geom), ht.height * S / ht.square])
    tg, tr = ht.t_grasp, ht.t_release
    tgt_g = np.array([*geom.square_xy(e.src), gz])
    tgt_r = np.array([*geom.square_xy(e.dst), gz + 0.004])
    d_g, d_r = tgt_g - p[tg], tgt_r - p[tr]
    t = np.arange(T)
    nb = max(1, int(blend_s * FPS))
    w_in = _smoothstep((t - (tg - nb)) / nb)
    w_out = 1.0 - _smoothstep((t - tr) / nb)
    a = np.clip((t - tg) / max(tr - tg, 1), 0.0, 1.0)
    delta = np.where((t < tg)[:, None], w_in[:, None] * d_g,
                     np.where((t <= tr)[:, None], (1 - a)[:, None] * d_g + a[:, None] * d_r,
                              w_out[:, None] * d_r))
    q = p + delta

    # 2. Clearance while the gripper is open: stay above anything nearby.
    z_hover = gz + hover
    def clear_open(idx, extra=()):
        for xy, h in list(obstacles) + list(extra):
            near = np.linalg.norm(q[idx, :2] - xy, axis=1) < 1.2 * S
            q[idx, 2] = np.where(near, np.maximum(q[idx, 2], max(h + 0.015, z_hover)), q[idx, 2])

    nf = min(int(funnel_s * FPS), tg)
    nl = int(lift_s * FPS)
    nf_r = min(int(funnel_s * FPS), max(1, tr - tg - nl - 1))
    piece_h = shape.height * k
    clear_open(np.arange(0, tg - nf), extra=[(tgt_g[:2], piece_h)])
    clear_open(np.arange(min(tr + nl, T - 1), T), extra=[(tgt_r[:2], piece_h)])

    # 3. Pre-grasp funnel: rise if low, move over the piece, descend vertically.
    p0 = q[tg - nf].copy()
    z_safe = max(p0[2], z_hover)
    q[tg - nf: tg + 1] = _polyline([p0, [*p0[:2], z_safe], [*tgt_g[:2], z_safe], tgt_g],
                                   [0.0, 0.2, 0.6, 1.0], nf + 1)
    # 4. Lift vertically, carry above every standing piece, pre-place funnel.
    lift_end = min(tg + nl, tr - nf_r - 1)
    q[tg: lift_end + 1] = _polyline([tgt_g, [*tgt_g[:2], carry_z]], [0.0, 1.0], lift_end - tg + 1)
    q[lift_end: tr - nf_r + 1, 2] = np.maximum(q[lift_end: tr - nf_r + 1, 2], carry_z)
    p1 = q[tr - nf_r].copy()
    q[tr - nf_r: tr + 1] = _polyline([p1, [*tgt_r[:2], max(p1[2], carry_z)], tgt_r], [0.0, 0.55, 1.0], nf_r + 1)
    # 5. Retreat vertically before rejoining the human path.
    r_end = min(tr + nl, T - 1)
    q[tr: r_end + 1] = _polyline([tgt_r, [*tgt_r[:2], z_hover]], [0.0, 1.0], r_end - tr + 1)

    # Yaw: the human's, held constant from the funnel through the carry.
    yaw = np.unwrap(_heading_to_yaw(ht.heading) * 2) / 2
    yaw_g, yaw_r = yaw[tg], yaw[tg]
    if yaw_fn is not None:
        yaw_g, yaw_r = yaw_fn(yaw_g, yaw_r)
    w_hold = np.clip((t - (tg - nf)) / max(nf, 1), 0, 1)
    yaw_seq = np.where(t < tg, (1 - w_hold) * yaw + w_hold * yaw_g,
                       np.where(t <= tr, yaw_g, (1 - w_out) * yaw + w_out * yaw_g))

    width = np.where((t >= tg) & (t <= tr), 0.0, pre_w)

    # Dwells at both keyframes: first let the arm *arrive* (it lags the
    # rate-limited target), then actuate the fingers and hold while they move.
    na, nd = int(arrive_s * FPS), int(dwell_s * FPS)
    idx = np.concatenate([np.arange(0, tg + 1), np.full(na + nd, tg), np.arange(tg + 1, tr + 1),
                          np.full(na + nd, tr), np.arange(tr + 1, T)])
    out = np.column_stack([q, yaw_seq, width])[idx]
    out[tg: tg + 1 + na, 4] = pre_w            # still open while arriving
    out[tg + 1 + na: tg + 1 + na + nd, 4] = 0.0
    j = tg + 1 + na + nd + (tr - tg)
    out[j - (tr - tg) - 0: j, 4] = 0.0         # carry closed
    out[j: j + na, 4] = 0.0                    # still closed while arriving
    out[j + na: j + na + nd, 4] = pre_w
    if trim:
        first = max(0, tg - nf - int(lead_s * FPS))
        last = min(len(out), j + na + nd + nl + int(tail_s * FPS))
        out, idx = out[first:last], idx[first:last]
    n = _n_steps(len(out), speed)
    res = _resample(out, n)
    res[:, 4] = np.where(res[:, 4] < 0.5 * pre_w, 0.0, pre_w)
    if return_source_index:  # dataset frame each Panda step was retargeted from
        return res, idx[np.round(np.linspace(0, len(idx) - 1, n)).astype(int)]
    return res
