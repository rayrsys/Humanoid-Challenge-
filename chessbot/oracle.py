"""Scripted pick-and-place for any chess move.

This is a *reference* (an upper bound and an env sanity check), not a source of
training data: every policy in this project is trained on retargeted human
demonstrations.
"""

from __future__ import annotations

import chess
import numpy as np

from .board import HAND_HALF_LENGTH, HAND_HALF_WIDTH, HAND_PALM_Z, PIECE_SHAPES

SAFE_Z = 0.14  # carry height: a hanging king clears a standing king
YAW_CANDIDATES = (0.0, np.pi / 2, np.pi / 4, -np.pi / 4)


def palm_clearance(env, xy, grasp_z: float, yaw: float, ignore=()) -> float:
    """Smallest gap (m) between the palm underside and any piece under its footprint.

    The Panda hand is 21 cm long along the finger axis, so when grasping low
    it sweeps over neighbouring squares; tall neighbours there collide.
    """
    k = env.geom.scale
    palm = grasp_z + HAND_PALM_Z
    along = np.array([np.sin(yaw), -np.cos(yaw)])  # finger closing axis in the board frame
    across = np.array([np.cos(yaw), np.sin(yaw)])
    gap = np.inf
    for sq, name in env.occupant.items():
        if sq in ignore:
            continue
        p, _ = env.piece_pose(name)
        rel = p[:2] - xy
        r = 0.0125 * k
        if abs(rel @ along) < HAND_HALF_LENGTH + r and abs(rel @ across) < HAND_HALF_WIDTH + r:
            gap = min(gap, palm - PIECE_SHAPES[name.split("_")[1]].height * k)
    return gap


def choose_yaw(env, move: chess.Move, grasp_z: float, margin: float = 0.004) -> float:
    """First candidate yaw whose palm clears neighbours at both pick and place."""
    a = env.geom.square_xy(move.from_square)
    b = env.geom.square_xy(move.to_square)
    ignore = (move.from_square, move.to_square)
    best, best_gap = YAW_CANDIDATES[0], -np.inf
    for yaw in YAW_CANDIDATES:
        gap = min(palm_clearance(env, a, grasp_z, yaw, ignore), palm_clearance(env, b, grasp_z, yaw, ignore))
        if gap >= margin:
            return yaw
        if gap > best_gap:
            best, best_gap = yaw, gap
    return best


def pick_place_waypoints(env, move: chess.Move | str, yaw: float | None = None):
    """List of (action, hold_steps) for moving the piece on move.from_square."""
    move = chess.Move.from_uci(move) if isinstance(move, str) else move
    shape = PIECE_SHAPES[env.occupant[move.from_square].split("_")[1]]
    k = env.geom.scale
    gz = shape.grasp_z * k
    pre = 2 * shape.grasp_radius * k + 0.024  # open just enough to clear the neck
    if yaw is None:
        yaw = choose_yaw(env, move, gz)
    a = env.geom.square_xy(move.from_square)
    b = env.geom.square_xy(move.to_square)

    def act(xy, z, w):
        return np.array([xy[0], xy[1], z, yaw, w])

    return [
        (act(a, SAFE_Z, pre), 0),
        (act(a, gz + 0.03, pre), 0),
        (act(a, gz, pre), 3),
        (act(a, gz, 0.0), 10),
        (act(a, SAFE_Z, 0.0), 0),
        (act(b, SAFE_Z, 0.0), 2),
        (act(b, gz + 0.004, 0.0), 4),
        (act(b, gz + 0.004, pre), 8),
        (act(b, SAFE_Z, pre), 0),
    ]


def run_waypoints(env, waypoints, tol: float = 0.003, max_steps: int = 80, on_step=None):
    """Drive the env through waypoints; returns the executed action trajectory."""
    actions = []
    for target, hold in waypoints:
        for _ in range(max_steps):
            obs = env.step(target)
            actions.append(target.copy())
            if on_step:
                on_step(obs)
            if np.linalg.norm(env.cmd[:3] - target[:3]) < 1e-6 and \
                    np.linalg.norm(obs["ee"][:3] - target[:3]) < tol:
                break
        for _ in range(hold):
            obs = env.step(target)
            actions.append(target.copy())
            if on_step:
                on_step(obs)
    return np.array(actions)


def oracle_mover(env, a: int, b: int, on_step=None):
    """``game.execute_move``-compatible mover backed by the scripted oracle."""
    run_waypoints(env, pick_place_waypoints(env, chess.Move(a, b)), on_step=on_step)


def play_move(env, move, on_step=None):
    move = chess.Move.from_uci(move) if isinstance(move, str) else move
    traj = run_waypoints(env, pick_place_waypoints(env, move), on_step=on_step)
    return env.check_move(move), traj
