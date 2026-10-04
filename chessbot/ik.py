"""Damped-least-squares IK for the Panda pinch point, with a posture null-space."""

from __future__ import annotations

import mujoco
import numpy as np

from .scene import POSTURE_QPOS

# Top-down grasp: hand z points down, fingers close along the rotated world y.
_TOP_DOWN = np.diag([1.0, -1.0, -1.0])


def grasp_rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ _TOP_DOWN


def yaw_of(mat: np.ndarray) -> float:
    """Inverse of :func:`grasp_rotation` for a (near) top-down orientation."""
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


class PandaIK:
    def __init__(self, model: mujoco.MjModel, site: str = "pinch"):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(site).id
        joints = [model.joint(f"joint{i}") for i in range(1, 8)]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dofadr = np.array([j.dofadr[0] for j in joints])
        self.lo = np.array([j.range[0] for j in joints])
        self.hi = np.array([j.range[1] for j in joints])
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, q0, pos, mat, iters: int = 40, tol: float = 2e-4, damping: float = 0.03,
              rot_weight: float = 0.5, posture_gain: float = 0.05) -> tuple[np.ndarray, float]:
        """Return joint angles reaching (pos, mat) and the final position error."""
        q = np.array(q0, dtype=float)
        quat_err = np.zeros(4)
        quat_cur = np.zeros(4)
        quat_tgt = np.zeros(4)
        mujoco.mju_mat2Quat(quat_tgt, np.asarray(mat, float).flatten())
        err_pos = np.inf
        for _ in range(iters):
            x, r = self.fk(q)
            e_p = pos - x
            mujoco.mju_mat2Quat(quat_cur, r.flatten())
            mujoco.mju_negQuat(quat_err, quat_cur)
            mujoco.mju_mulQuat(quat_err, quat_tgt, quat_err)
            e_r = np.zeros(3)
            mujoco.mju_quat2Vel(e_r, quat_err, 1.0)
            err_pos = float(np.linalg.norm(e_p))
            if err_pos < tol and np.linalg.norm(e_r) < 10 * tol:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
            jac = np.vstack([self._jp[:, self.dofadr], rot_weight * self._jr[:, self.dofadr]])
            err = np.concatenate([e_p, rot_weight * e_r])
            jjt = jac @ jac.T + damping**2 * np.eye(6)
            inv = np.linalg.solve(jjt, np.eye(6))
            dq = jac.T @ inv @ err
            null = np.eye(7) - jac.T @ inv @ jac
            dq += null @ (posture_gain * (POSTURE_QPOS - q))
            q = np.clip(q + dq, self.lo, self.hi)
        return q, err_pos
