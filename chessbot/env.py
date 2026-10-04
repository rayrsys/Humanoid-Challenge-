"""Chess manipulation environment: a Panda moving physical pieces on a board.

Action (5,), all in the board frame (see ``board.py``):
    [x, y, z, yaw, width]  pinch-point target in metres, gripper yaw in rad,
                           finger opening in metres (0 = closed, 0.08 = fully open)

Commanding a width smaller than the object squeezes it, exactly as a human
pinch does; this keeps the action space directly comparable to the
thumb-index aperture measured by hand tracking.

The commanded target is rate-limited (like a real Cartesian controller), then
converted to joint targets by IK and tracked by Menagerie's position servos.

Grasp stabiliser: MuJoCo's soft contacts let a pinched piece creep ~1 cm along
the finger pads during fast transport (independent of friction; see README).
So once *both* pads are squeezing a piece, it is welded to the hand at its
current offset, and released as soon as the fingers open. Approach, contact,
collisions with other pieces, placement and release all remain physical.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import mujoco
import numpy as np

from .board import BoardGeometry
from .ik import PandaIK, grasp_rotation, yaw_of
from .scene import HOME_QPOS, POOL, build_model, pool_names

CONTROL_HZ = 20
GRIPPER_OPEN = 0.04  # metres per finger
MAX_WIDTH = 2 * GRIPPER_OPEN


@dataclass
class MoveResult:
    success: bool
    placed_square: str | None
    placement_error: float  # metres from the target square centre
    upright: bool
    disturbed: list  # names of other pieces knocked off their squares
    released: bool

    def as_dict(self) -> dict:
        return dict(success=self.success, placed_square=self.placed_square,
                    placement_error=self.placement_error, upright=self.upright,
                    disturbed=self.disturbed, released=self.released)


class ChessEnv:
    def __init__(self, geom: BoardGeometry | None = None, cameras=("front", "wrist"),
                 image_size=(224, 224), max_speed: float = 0.4, max_yaw_rate: float = 3.0):
        self.geom = geom or BoardGeometry()
        self.model = build_model(self.geom)
        self.data = mujoco.MjData(self.model)
        self.ik = PandaIK(self.model)
        self.cameras = tuple(cameras)
        self.image_size = image_size
        self.max_step = max_speed / CONTROL_HZ
        self.max_yaw_step = max_yaw_rate / CONTROL_HZ
        self.n_substeps = int(round(1.0 / (CONTROL_HZ * self.model.opt.timestep)))
        self._renderer = None

        self.pinch = self.model.site("pinch").id
        self.names = pool_names()
        self.body = {n: self.model.body(n).id for n in self.names}
        self.qadr = {n: self.model.joint(f"{n}_free").qposadr[0] for n in self.names}
        self.dadr = {n: self.model.joint(f"{n}_free").dofadr[0] for n in self.names}
        self.finger_qadr = [self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        self.hold_eq = {n: self.model.equality(f"{n}_hold").id for n in self.names}
        self.hand = self.model.body("hand").id
        self.fingers = (self.model.body("left_finger").id, self.model.body("right_finger").id)
        self.held: str | None = None
        self.board = chess.Board()
        self.occupant: dict[int, str] = {}  # square -> pool body name
        self._next_slot = 0
        self.cmd = np.zeros(5)
        self.t = 0

    # ------------------------------------------------------------------ setup
    def reset(self, fen: str = chess.STARTING_FEN, jitter: float = 0.0, seed: int | None = None,
              settle_steps: int = 10):
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.eq_active[:] = 0
        self.held = None
        d.qpos[self.ik.qadr] = HOME_QPOS
        d.qpos[self.finger_qadr] = GRIPPER_OPEN
        d.ctrl[:7] = HOME_QPOS
        d.ctrl[7] = 255.0

        self.board = chess.Board(fen)
        self.occupant = {}
        free = {(c, s): [n for n in self.names if n.startswith(f"{c}_{s}_")] for c in "wb" for s in POOL}
        for sq, piece in self.board.piece_map().items():
            key = ("w" if piece.color else "b", piece.symbol().lower())
            if not free[key]:
                raise ValueError(f"FEN needs more {key} pieces than the pool has")
            name = free[key].pop(0)
            xy = self.geom.square_xy(sq) + rng.uniform(-jitter, jitter, 2)
            self._place(name, xy, yaw=rng.uniform(-0.3, 0.3) if jitter else 0.0)
            self.occupant[sq] = name
        self._next_slot = 0
        for name in [n for lst in free.values() for n in lst]:
            self._park(name)

        mujoco.mj_forward(m, d)
        x, r = self._pinch_pose()
        self.cmd = np.concatenate([self.geom.to_board(x), [yaw_of(r), MAX_WIDTH]])
        for _ in range(settle_steps):
            self.step(self.cmd)
        self.t = 0
        return self.observe()

    def _place(self, name: str, xy, yaw: float = 0.0, z: float = 0.0005):
        a = self.qadr[name]
        self.data.qpos[a:a + 3] = self.geom.to_world(np.array([xy[0], xy[1], z]))
        self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[self.dadr[name]:self.dadr[name] + 6] = 0

    def _park(self, name: str):
        """Graveyard: rows of 10 beside the board on the h-file side, one slot per piece."""
        slot, self._next_slot = self._next_slot, self._next_slot + 1
        side = 4 * self.geom.square + self.geom.border + 0.04
        x = (slot % 10 - 4.5) * 0.045
        y = -(side + 0.045 * (slot // 10))
        self._place(name, (x, y), z=-self.geom.origin[2] + 0.0005)

    # --------------------------------------------------------------- control
    def step(self, action):
        a = np.asarray(action, dtype=float).copy()
        a[2] = max(a[2], 0.012)  # keep fingertips off the board surface
        a[4] = np.clip(a[4], 0.0, MAX_WIDTH)
        d_pos = a[:3] - self.cmd[:3]
        n = np.linalg.norm(d_pos)
        if n > self.max_step:
            d_pos *= self.max_step / n
        d_yaw = (a[3] - self.cmd[3] + np.pi) % (2 * np.pi) - np.pi
        d_yaw = np.clip(d_yaw, -self.max_yaw_step, self.max_yaw_step)
        self.cmd = np.concatenate([self.cmd[:3] + d_pos, [self.cmd[3] + d_yaw, a[4]]])

        q0 = self.data.qpos[self.ik.qadr]
        q, _ = self.ik.solve(q0, self.geom.to_world(self.cmd[:3]), grasp_rotation(self.cmd[3]))
        self.data.ctrl[:7] = q
        self.data.ctrl[7] = 255.0 * self.cmd[4] / MAX_WIDTH
        self._update_grasp()
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self.t += 1
        return self.observe()

    def _finger_width(self) -> float:
        return float(self.data.qpos[self.finger_qadr].sum())

    def _squeezed_piece(self) -> str | None:
        """Piece in contact with both finger pads, if any."""
        m, d = self.model, self.data
        touched: dict[int, set] = {}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for f, o in ((b1, b2), (b2, b1)):
                if f in self.fingers:
                    touched.setdefault(o, set()).add(f)
        for name in self.names:
            if len(touched.get(self.body[name], ())) == 2:
                return name
        return None

    def _update_grasp(self):
        m, d = self.model, self.data
        width = self._finger_width()
        if self.held is not None:
            if self.cmd[4] > width + 0.003:  # fingers commanded open: release
                d.eq_active[self.hold_eq[self.held]] = 0
                self.held = None
            return
        if self.cmd[4] < width - 0.002:  # commanded tighter than achieved: squeezing something
            name = self._squeezed_piece()
            if name is None:
                return
            eq, b = self.hold_eq[name], self.body[name]
            r_hand = d.xmat[self.hand].reshape(3, 3)
            q_inv, q_rel = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(q_inv, d.xquat[self.hand])
            mujoco.mju_mulQuat(q_rel, q_inv, d.xquat[b])
            m.eq_data[eq, :3] = 0.0
            m.eq_data[eq, 3:6] = r_hand.T @ (d.xpos[b] - d.xpos[self.hand])
            m.eq_data[eq, 6:10] = q_rel
            m.eq_data[eq, 10] = 1.0
            d.eq_active[eq] = 1
            self.held = name

    # ----------------------------------------------------------- observation
    def _pinch_pose(self):
        return self.data.site_xpos[self.pinch].copy(), self.data.site_xmat[self.pinch].reshape(3, 3).copy()

    def ee_state(self) -> np.ndarray:
        """[x, y, z, yaw, gripper_opening] of the pinch point, board frame."""
        x, r = self._pinch_pose()
        width = float(self.data.qpos[self.finger_qadr].sum())
        return np.concatenate([self.geom.to_board(x), [yaw_of(r), width]])

    def observe(self, render: bool = True) -> dict:
        obs = {
            "ee": self.ee_state(),
            "qpos": self.data.qpos[self.ik.qadr].copy(),
            "cmd": self.cmd.copy(),
        }
        if render and self.cameras:
            obs["images"] = {c: self.render(c) for c in self.cameras}
        return obs

    def render(self, camera: str = "front", size=None) -> np.ndarray:
        h, w = size or self.image_size
        if self._renderer is None or (self._renderer.height, self._renderer.width) != (h, w):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, h, w)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    # ------------------------------------------------------------ evaluation
    def piece_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.body[name]
        return self.geom.to_board(self.data.xpos[b]), self.data.xmat[b].reshape(3, 3)

    def piece_on(self, sq: str | int) -> str | None:
        sq = chess.parse_square(sq) if isinstance(sq, str) else sq
        return self.occupant.get(sq)

    def _in_square(self, name: str, sq: int, tol: float) -> tuple[bool, float, bool]:
        p, r = self.piece_pose(name)
        err = float(np.linalg.norm(p[:2] - self.geom.square_xy(sq)))
        upright = bool(r[2, 2] > 0.95)
        return err < tol, err, upright

    def check_move(self, move: chess.Move | str, tol_frac: float = 0.3) -> MoveResult:
        """Did the piece on ``move.from_square`` physically end up on ``to_square``?

        Success = centred within ``tol_frac`` of a square, upright, gripper no
        longer touching it, and no other piece pushed off its square.
        """
        move = chess.Move.from_uci(move) if isinstance(move, str) else move
        name = self.occupant[move.from_square]
        tol = tol_frac * self.geom.square
        ok, err, upright = self._in_square(name, move.to_square, tol)
        p, _ = self.piece_pose(name)
        placed = self.geom.xy_to_square(p[:2])
        released = not self._touching_gripper(name) and p[2] < 0.01
        disturbed = []
        for sq, other in self.occupant.items():
            if other == name or sq == move.to_square:
                continue
            o_ok, _, o_up = self._in_square(other, sq, tol)
            if not (o_ok and o_up):
                disturbed.append(other)
        return MoveResult(bool(ok and upright and released and not disturbed),
                          chess.square_name(placed) if placed is not None else None,
                          err, upright, disturbed, released)

    def _touching_gripper(self, name: str) -> bool:
        b = self.body[name]
        fingers = {self.model.body("left_finger").id, self.model.body("right_finger").id,
                   self.model.body("hand").id}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 == b and b2 in fingers) or (b2 == b and b1 in fingers):
                return True
        return False

    def commit_move(self, move: chess.Move | str):
        """Update the logical board after a physical move (captures are cleared first)."""
        move = chess.Move.from_uci(move) if isinstance(move, str) else move
        name = self.occupant.pop(move.from_square)
        self.occupant[move.to_square] = name
        self.board.push(move)

    def remove_piece(self, sq: int | str):
        """Teleport a captured piece to the graveyard (captures are not manipulated)."""
        sq = chess.parse_square(sq) if isinstance(sq, str) else sq
        name = self.occupant.pop(sq)
        self._park(name)
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
