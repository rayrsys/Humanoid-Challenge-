---
license: cc-by-4.0
task_categories:
- robotics
tags:
- lerobot
- humanoid
- unitree-g1
- dexterous-manipulation
- chess
- isaac-sim
pretty_name: Humanoid Chess FB32 (all five pieces, left arm)
---

# Humanoid Chess FB32 — all five pieces, left arm

338 teleoperated demonstrations of a simulated **Unitree G1 humanoid** moving chess pieces with its
**left three-finger hand** on a full 8x8 board in **NVIDIA Isaac Sim 5.1**. Part of the FB32 benchmark
(masters research, University of the Witwatersrand).

- **Pieces:** king, queen, rook, bishop, knight (both colours), ~68 demos each; 34 distinct source-to-destination moves per piece, each collected twice. Only successful demos are included.
- **Collection:** VR teleoperation (Meta Quest 3 over NVIDIA CloudXR), 50 Hz, Pink IK on the wrists, direct hand-joint targets.
- **Size:** 338 episodes, LeRobot v3.0 format.

## Features
| key | shape | description |
|---|---|---|
| observation.images.head | 240x320x3 video | head camera |
| observation.images.left_wrist | 240x320x3 video | left wrist camera |
| observation.state | 28 | both wrist positions (3+3) and quaternions (4+4), 14 hand joint angles |
| action | 28 | absolute wrist pose targets (both arms) + 14 hand joint targets |
| task | text | e.g. "Move the white knight from h1 to f2." |

The destination square shows a flat coloured marker whose colour encodes the piece type (king yellow,
queen violet, rook red, bishop magenta, knight cyan).

## Loading
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("Raysolo/fb32-v03-all5-left")
```

## Use in the FB32 benchmark
ACT, Diffusion Policy, SmolVLA and GR00T N1.6/N1.7 were trained on this data and evaluated on four suites
(trained moves, identifying the right piece among two, an unfamiliar pawn as distractor, an unfamiliar pawn as
target).
