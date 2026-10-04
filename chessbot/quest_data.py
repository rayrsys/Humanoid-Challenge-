"""Loader for the Quest 3 teleoperation datasets (LeRobot v3, G1 + Dex3 hands).

Each dataset (``fb32-v03-all5-{right,left}``) holds single chess moves
demonstrated by teleoperating a Unitree G1 humanoid with a Meta Quest 3.
The 28-D state / action layout (from ``provenance.json``):

    [0:3]  left wrist position      [3:7]   left wrist quaternion (w, x, y, z)
    [7:10] right wrist position     [10:14] right wrist quaternion (w, x, y, z)
    [14:28] Dex3 hand joints, interleaved per hand in groups of three:
            left = 14,15,16,20,21,22,26   right = 17,18,19,23,24,25,27

Two grasp styles appear in the demos (checked against the head camera):
  * power grasp (knights, bishops, rooks): two finger-flexion joints
    (left 20,21 / right 23,24) swing from ~0 to ~1.6 rad;
  * precision pinch (most kings and queens): thumb + index fingertips on the
    crown; flexion barely moves but the thumb joints do.
So "closure" is measured over all 7 hand joints relative to the episode's
initial open pose, and normalised per episode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 50
ARM_SLICES = {
    # The left Dex3 hand is mirrored: its flexion joints go negative when closing.
    "left": dict(pos=slice(0, 3), quat=slice(3, 7), hand=[14, 15, 16, 20, 21, 22, 26], grip=[20, 21], sign=-1.0),
    "right": dict(pos=slice(7, 10), quat=slice(10, 14), hand=[17, 18, 19, 23, 24, 25, 27], grip=[23, 24], sign=1.0),
}


@dataclass
class Episode:
    index: int
    episode_id: str
    arm: str
    piece: str            # "knight", ...
    colour: str           # "white" / "black" (side to move in the FEN)
    src: str | None
    dst: str | None
    fen: str | None
    task: str
    pos: np.ndarray       # (T, 3) wrist position, robot frame
    quat: np.ndarray      # (T, 4) wrist orientation (w, x, y, z)
    hand: np.ndarray      # (T, 7) Dex3 joints of the active hand
    grip: np.ndarray      # (T,) mean flexion of the two main finger joints
    act_pos: np.ndarray   # commanded wrist position (teleop target)
    act_quat: np.ndarray
    act_grip: np.ndarray

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.pos)) / FPS

    @property
    def labelled(self) -> bool:
        return bool(self.src and self.dst and self.fen)


def default_data_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def load_dataset(name: str, root: str | Path | None = None) -> list[Episode]:
    import pandas as pd

    d = Path(root or default_data_root()) / name
    df = pd.read_parquet(sorted((d / "data").rglob("*.parquet")))
    prov = json.loads((d / "provenance.json").read_text())["episodes"]
    tasks = pd.read_parquet(d / "meta" / "tasks.parquet")
    task_of = {int(v): k for k, v in tasks["task_index"].items()} if "task_index" in tasks else {}

    episodes = []
    for ep_idx, g in df.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index")
        p = prov[str(ep_idx)]
        arm = p.get("arm") or ("left" if "left" in name else "right")
        sl = ARM_SLICES[arm]
        s = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
        a = np.stack(g["action"].to_numpy()).astype(np.float64)
        fen = p.get("starting_fen")
        colour = ("white" if fen.split()[1] == "w" else "black") if fen else "unknown"
        episodes.append(Episode(
            index=int(ep_idx), episode_id=p["episode_id"], arm=arm, piece=p["piece_type"], colour=colour,
            src=p.get("source_square"), dst=p.get("destination_square"), fen=fen,
            task=task_of.get(int(g["task_index"].iloc[0]), ""),
            pos=s[:, sl["pos"]], quat=s[:, sl["quat"]], hand=s[:, sl["hand"]], grip=sl["sign"] * s[:, sl["grip"]].mean(1),
            act_pos=a[:, sl["pos"]], act_quat=a[:, sl["quat"]], act_grip=sl["sign"] * a[:, sl["grip"]].mean(1),
        ))
    return episodes


def closure(hand: np.ndarray, n_open: int = 15) -> np.ndarray:
    """Distance of the 7 hand joints from the episode's initial (open) pose."""
    return np.linalg.norm(hand - np.median(hand[:n_open], axis=0), axis=1)


def grasp_events(hand: np.ndarray, close: float = 0.55, open_: float = 0.35, min_hold: int = 15,
                 min_closure: float = 0.15):
    """(grasp_idx, release_idx) of the main grasp, by hysteresis on normalised closure.

    Returns the longest closed interval; None if the hand never closes.
    """
    c = closure(hand)
    if c.max() < min_closure:
        return None
    grip = c / c.max()
    closed = False
    start, best = None, None
    for i, g in enumerate(grip):
        if not closed and g > close:
            closed, start = True, i
        elif closed and g < open_:
            closed = False
            if i - start >= min_hold and (best is None or i - start > best[1] - best[0]):
                best = (start, i)
    if closed and len(grip) - start >= min_hold and (best is None or len(grip) - start > best[1] - best[0]):
        best = (start, len(grip) - 1)
    return best
