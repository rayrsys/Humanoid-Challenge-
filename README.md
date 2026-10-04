# From a humanoid's hands to a Panda's gripper: chess moves learned from my Quest 3 teleoperation

I collected **676 chess-move demonstrations with a Meta Quest 3**, teleoperating a simulated
Unitree G1 humanoid with dexterous Dex3 hands (both arms, five piece types). The motion is real:
my own hands, tracked by the headset; the robot it drove was simulated. This project uses that data to
drive a **Franka Panda with a parallel-jaw gripper** in a MuJoCo chess simulator, and to train
policies that execute language instructions such as *"Move the black knight from b8 to c6."*

The interesting part is the gap between the two embodiments: a five-fingered humanoid hand that
pinches crowns and wraps around knights, versus a two-finger gripper with a 21 cm palm. The
pipeline closes that gap step by step and measures each step.

| Taken from my recordings | Adapted for the Panda |
|---|---|
| every task: the instruction, starting position and move | the last ~0.6 s before each grasp and release: hover, vertical descent, pause, close/open (a gripper cannot copy a Dex3 contact) |
| where the board was (fitted from my wrist poses alone) | clearance above neighbouring pieces while the gripper is open |
| the hand's path, timing and speed profile, when it grasps and releases, and the wrist yaw | speed capped at the Panda's limits, and a short transit from wherever the arm is to the start of the motion |

The policies train only on my demonstrations (retargeted, plus board-symmetric copies of the same
motions), and the full game replays them. The scripted pick-and-place in `chessbot/oracle.py` is
only used to test the simulator.

<p align="center"><img src="docs/fig_teleop_vs_panda.png" width="760"></p>

**Headline results**

| | |
|---|---|
| Board pose recovered from the demos alone | square size **39.97 / 40.43 mm** (left/right arm, true 40 mm), median fingertip error **≈5 mm** |
| Naive retargeting (copy the hand path) | **27.7 %** of 675 demos succeed on the Panda |
| Object-centric retargeting (keeps the human's timing, path and yaw) | **99.9 %** succeed (674 / 675) |
| Board-symmetry augmentation | 499 human demos → **3 195** sim-verified training episodes (augmented copies succeed as often as originals: 99.9 % vs 99.3 %) |
| Policy trained on them, closed loop | **85 % / 78 % / 70 %** on seen tasks / unseen square pairs / squares never seen in training (flat square embeddings: 0 % on unseen squares; no augmentation: ≤ 18 %) |
| Whole game | Morphy's *Opera Game* (1858), 33 plies played with my recorded hand motions: **34 / 34** piece moves, **no interventions**, on each of 10 random choices of which demo is replayed per move |

---

## 1. The data I collected

| | |
|---|---|
| Hardware | Meta Quest 3 hand tracking → simulated Unitree G1 (Dex3-1 hands), one arm at a time |
| Datasets | [`Raysolo/fb32-v03-all5-right`](https://huggingface.co/datasets/Raysolo/fb32-v03-all5-right), [`Raysolo/fb32-v03-all5-left`](https://huggingface.co/datasets/Raysolo/fb32-v03-all5-left) (LeRobot v3) |
| Size | 676 episodes (338 per arm), 180 k frames at 50 Hz, ≈ 5 s per move |
| Tasks | knight, bishop, rook, queen, king; white and black; one move per episode on a sparse board, with a language instruction and the starting FEN |
| Signals | wrist pose of both arms, 14 Dex3 finger joints, head and wrist camera video |

Two grasp styles show up in the data (checked against the head camera): a **power grasp** that
closes the whole hand (knights, bishops, rooks), and a **thumb-index pinch** on the crown (most
kings and queens). That difference matters later.

## 2. Recovering the board from the demonstrations

The recordings contain the wrist pose but not where the board was. Every episode does say which
two squares were touched, though, and the hand tells us when. Assuming a level board and a fixed
fingertip offset `o` in the wrist frame, at every grasp and release

```
p_wrist[:2] + (R_wrist · o)[:2] = origin + A · [file, rank]
```

which is **linear** in the 9 unknowns (origin, the 2×2 board axes `A`, and `o`): 1 350 events,
one least-squares solve (`chessbot/calibration.py`). Grasp timing is bootstrapped from the
unambiguous power grasps, then every episode is re-segmented by its closest low approach to the
labelled squares and the fit is repeated.

<p align="center"><img src="docs/fig_calibration.png" width="760"></p>

* The fitted square is **39.97 mm** (right arm) and **40.43 mm** (left); the board is 32 cm.
* The fitted fingertip offset is (14, ±4, 0.6) cm in the wrist frame, **mirrored** between the
  two hands, as it physically should be.
* The two arms were fitted independently and place the same board within 12 mm of each other.
* Grasp heights come out ordered by piece: rook 19 mm < bishop 23 < knight 25 < queen 29 < king 36.

## 3. Cross-embodiment retargeting

Both robots play on a chessboard, so the shared frame is the **board**: each demonstration is
expressed as a continuous (file, rank, height) trajectory of the fingertip point, then re-embodied
on the Panda's board (`chessbot/retarget.py`).

<p align="center"><img src="docs/fig_retargeting.png" width="700"></p>

| | Naive | Object-centric |
|---|---|---|
| What it does | scale the fingertip path to the Panda board, gripper yaw from the wrist heading, gripper width from hand closure | keep the human's **timing, path shape and yaw**; pin the grasp/release keyframes to the piece; Panda-specific contact funnels |
| Success (675 episodes) | **27.7 %** | **99.9 %** |
| Failures | wrong square (239), off-centre (151), dropped in transit (82), tipped over (8), stuck to a finger (7) | off-centre (1) |

The naive version fails for reasons that are informative about the embodiment gap:
a dexterous hand approaches a piece diagonally from the side, which a parallel gripper does with a
finger *into* the piece; the human pinch relaxes during transport (fine for a humanoid hand,
fatal for a width-controlled gripper); and the hand's closure signal does not say *when* a pinch
releases.

How the object-centric retargeter got from 54 % to 95 % (first 60 episodes, in order):

| Change | Success |
|---|---|
| Keyframes pinned to the piece, MimicGen-style offset blending | 54 % |
| + vertical pre-grasp / pre-place funnels and clearance above standing pieces | 63 % |
| + "arrive" dwell before actuating the fingers (the arm lags its rate-limited target) | 95 % |

Everything a human contributes survives: *when* to grasp and release, the transport path and
speed profile, and the wrist yaw. Only the last few centimetres of contact are adapted to the
gripper.

### The last 5 %: simulator bugs, not retargeting

At 94.8 % I traced every remaining failure step by step (contacts, finger joints, joint torques).
Almost none were retargeting problems:

| Fix (all 675 episodes, in the order I made them) | Success | What was wrong |
|---|---|---|
| starting point | 94.8 % | |
| keep the fingers coupled | 96.3 % | resetting the grasp welds also switched off Menagerie's equality that couples the two fingers, so one finger could stay pinned to a piece while the other opened, and carry it away |
| cap speed at the Panda's limits | 95.3 % | a human hand is faster than the arm; the rate limiter then cut corners and came down onto tall pieces before the hand was over them (13 → 0 such failures), but more placements on rank 1 now hit the next bug (5 → 20) |
| warm-start IK from the commanded joints | 96.7 % | near rank 1 the shoulder is close to a singularity (joints 1 and 3 aligned); solving from the measured state fed the arm's own tracking error back into its target |
| ramp joint setpoints over each 50 ms step | 99.3 % | stepping them saturated the 87 Nm shoulder joints, which disables the integrator's implicit damping: the pair chattered and placements landed ~2 cm off |
| hover above the tallest neighbour | 99.9 % | the open fingers could dip beside a neighbour before the pre-grasp funnel |
| transit from the arm's current pose | 99.9 % | the first target of a move could be 50 cm from where the last one ended (matters in the game, not in single-move replays) |

Because the order of fixes matters, I also removed each one from the final system on its own
(`scripts/ablate_fixes.py`):

| Final system without… | Replay success (675) | Failures |
|---|---|---|
| nothing removed | 99.9 % | off-centre 1 |
| setpoint ramp | 97.3 % | off-centre 14, dropped 3, wrong square 1 |
| speed cap | 97.2 % | dropped 10, off-centre 5, tipped over 3, stuck to a finger 1 |
| neighbour clearance | 99.4 % | dropped 3, off-centre 1 |
| start transit | 99.9 % | off-centre 1; in the full game 3 of 10 seeds then need an intervention (337 / 340 moves) |
| IK warm start | 100 % | none |
| finger coupling | 100 % | none |

So two fixes are redundant once the others are in: the warm start only mattered while setpoints
were stepped, and the coupling bug only carried pieces off after a grasp had already gone wrong
for another reason. I kept both, as they are what the real hardware does (the Panda's fingers are
mechanically coupled).


## 4. The simulator

There is no chessboard in LIBERO, and chess needs millimetre-level placement on 64 named
squares, so I built a small MuJoCo environment (`chessbot/env.py`, `chessbot/scene.py`):
Menagerie's Panda, a 40 cm board with 5 cm squares, a pool of 34 primitive pieces loaded from any
FEN, a damped-least-squares IK on the fingertip pinch point, and a 20 Hz absolute Cartesian
action `[x, y, z, yaw, width]`. A move counts as successful only if the piece ends centred on the
target square, upright, released, with **no other piece disturbed**.

Designing for the Panda hand forced some decisions:

* **Piece geometry follows the hand.** The finger pads span 17 mm and the palm sits 37 mm above
  the pinch point, so every piece has a straight grip band at its grasp height and its top within
  33 mm of it. The first king design was pressed into the board by the palm.
* **Yaw matters on a crowded board.** The palm is 21 cm long along the finger axis; picking a
  bishop next to a king means rotating the gripper so the palm clears it (`chessbot/oracle.py`).
* **Controller details matter.** IK is warm-started from the previous command and joint
  setpoints are ramped across the control period, as the real Panda's 1 kHz controller does; both
  fixed failures that looked like retargeting problems (see above).
* **Grasp stabiliser (a disclosed simplification).** Menagerie's fingertips have small "bump"
  colliders whose corners wedge cylinders sideways; I removed them. Even with flat pads, MuJoCo's
  soft contacts let a pinched piece creep ~1 cm along the pads during fast transport, independent of
  friction (tested: noslip solver, cone type, condim, friction, contact stiffness). So once both
  pads squeeze a piece it is welded to the hand at its current offset, and released when the
  fingers open. Approach, contact, collisions, placement and release stay physical.

## 5. Multiplying demonstrations with the board's symmetries

Translating a move (src + v, dst + v), or mirroring files, keeps it legal for its piece, and
because the human trajectory lives in board coordinates the whole motion moves with it
(`chessbot/augment.py`). Each demo was re-anchored to 4 random symmetric copies and every copy
was replayed on the Panda; only successful rollouts were kept.

* 3 375 rollouts → 3 366 successful; 3 195 in the training split (499 human + 2 696 augmented).
* Augmented copies succeed at the same rate as the originals (99.9 % vs 99.3 %), which is a
  useful check that the re-anchoring preserves motion quality.

The training rollouts differ from the replays in two ways, both removing something the
instruction cannot tell a policy:

* **They start at the pre-grasp funnel**, not a second earlier. Where my hand rested before
  reaching was on average 18 cm from the piece, and off the board in over half the demos.
* **The sign of the gripper yaw is canonicalised.** Which hand demonstrated decides it: left-hand
  demos grasp at about −0.9 rad, right-hand ones at +0.7, and mirrored copies flip it. Within one
  instruction the grasp yaw varied by 1.36 rad (median). Episodes that grasp at negative yaw have
  their whole yaw track mirrored, keeping the size and timing of my wrist turn.

**Held-out evaluation** (`chessbot/splits.py`): six squares (b6, c3, d7, e2, f5, g4) never
appear in training, as source or destination, and ~10 % of (piece, source, destination)
combinations are held out among the rest.

## 6. Policies

**Compact task-conditioned policy** (`chessbot/policy.py`, trains on a laptop CPU in minutes).
The instruction is parsed into piece, colour, source and destination; the network sees the
end-effector state and predicts 10-step chunks of Cartesian commands (receding horizon). Squares
are encoded either **factorised** (file embedding + rank embedding) or **flat** (one embedding per
square).

<p align="center"><img src="docs/fig_policies.png" width="760"></p>

60 tasks per split, the same seeded tasks and starting positions for every model, one training
seed each (with n = 60 the standard error is about ±6 points):

| Policy | Training episodes | Seen tasks | Unseen square pairs | Unseen squares |
|---|---|---|---|---|
| *Retargeted human replay (reference, not a policy)* | | *100 %* | *100 %* | *100 %* |
| **Augmented · factorised squares** | 3 195 | **85.0 %** | **78.3 %** | **70.0 %** |
| Augmented · flat squares | 3 195 | 60.0 % | 43.3 % | 0.0 % |
| Human only · factorised squares | 499 | 16.7 % | 0.0 % | 3.3 % |
| Human only · flat squares | 499 | 18.3 % | 0.0 % | 0.0 % |

Two things stand out. **Augmentation is what makes learning work**: 499 human demonstrations, one
per move, are far too few for a policy to learn where 64 squares are, while their symmetric copies
are enough. And **the square encoding decides generalisation**: with one embedding per square, a
square never seen in training is simply unknown (0 %); composing it from a seen file and a seen
rank gets 70 %.

<p align="center"><img src="docs/policy_in_action.gif" width="560"><br>
<em>The factorised policy given only the instruction and the gripper state: a seen task, an unseen
square pair, and two moves involving squares never seen in training (d7, e2). Inset: wrist camera.</em></p>

**SmolVLA.** `notebooks/smolvla_colab.ipynb` renders the training rollouts into a LeRobot dataset
(front + wrist cameras, 256×256) and fine-tunes `lerobot/smolvla_base` on a Colab GPU, then runs
the same closed-loop evaluation from images and the raw instruction (`scripts/eval_smolvla.py`).

## 7. Playing a whole game with my hand motions

`scripts/play_game.py --player human` plays Morphy's *Opera Game* (Paris 1858, 33 plies, captures,
queenside castling, a queen sacrifice and mate). For every move it retrieves one of my
demonstrations with the same piece and move vector (or its mirror image), re-anchors it to the
game's squares and executes it with the object-centric retargeter. Pawns are not in my data, so
they borrow the motion of a piece making the same step. Captured pieces are removed by the
environment rather than by the robot.

<p align="center"><img src="docs/opera_game.gif" width="560"></p>

| | |
|---|---|
| Plies | 33 (one castling, twelve captures) |
| Single-piece moves succeeding unassisted | 34 / 34 |
| Interventions (a piece snapped back onto its square so the game can continue) | 0 |
| Same game with 10 different random choices of which demo to replay per move | 340 / 340 moves, 0 interventions |

The first version needed 6 interventions, all in the crowded parts of the game: neighbours
knocked by an open finger, a rook carried off on one finger, misses on the back rank. Every one
traced back to the simulator and controller bugs above. My demonstrations were recorded on
sparse boards with three pieces, so a full board is still the hardest setting for motions that
were never meant to thread between neighbours.

## What worked, what didn't

**Worked**
* Treating the **board as the shared frame** between embodiments: calibration, retargeting,
  augmentation and evaluation all live in it.
* Recovering the board by **linear least squares** with the fingertip offset as an unknown.
* **Object-centric retargeting**: keeping the human's timing and path, adapting only contact.
* **Symmetry augmentation verified in simulation**, and **factorised square embeddings** for
  generalising to squares never seen in training.

**Didn't work, or only partly**
* Grasp detection from finger closure alone: it fails on pinch grasps (the fingers barely move
  and relax during the carry). The task labels fixed it.
* Naive retargeting (27.7 %), for the reasons above.
* Pure contact physics for carrying pieces: see the grasp stabiliser note.
* The first object-centric version opened the gripper while the arm was still moving (54 %).
* Trusting my simulator: the last 5 % of retargeting failures, and every intervention in the full
  game, were bugs in the finger coupling and the arm controller, not in the method. Tracing
  individual failures (contacts, joint torques) found them; tuning the retargeter would not have.
* Crowded full boards are harder than the sparse boards in the demos; the game script counts
  every intervention (now zero, but the margin around neighbours is a few millimetres).
* Training the policy on the replays as they were. Once the simulator was fixed, the factorised
  policy fell to 33 % on seen tasks, against 85 % now. Rollouts began wherever my hand happened
  to rest, and the grasp yaw had two modes the instruction cannot distinguish (which hand
  demonstrated), so a regression policy averaged them and approached at a yaw no demo used. A
  policy that can represent several modes (diffusion or a discretised action head) would be the
  principled fix; canonicalising the data was the quick one. (Before the simulator fixes, the
  same recipe gave 77 %; I suspect the buggy, uncoupled fingers forgave misaligned grasps, but did
  not verify it.)

**Limitations**
* The compact policy reads a parsed instruction and proprioception, not images; SmolVLA is the
  vision-language version.
* Captures, castling and promotion are decomposed into single-piece moves; captured pieces are
  removed by the environment.
* All results are in simulation: one MuJoCo Panda, one board geometry.

## Reproduce

```bash
git clone https://github.com/rayrsys/Humanoid-Challenge-.git && cd Humanoid-Challenge-
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,figures,dev]"
python scripts/download_data.py                     # teleop parquet + metadata (~50 MB)
export MUJOCO_GL=egl                                # osmesa on headless CPU, glfw on a desktop

python scripts/calibrate_board.py                   # board pose from the demos -> calib/
python scripts/replay.py --mode naive               # retargeting baseline
python scripts/replay.py --mode object_centric
python scripts/ablate_fixes.py                      # leave-one-out over the simulator/controller fixes
python scripts/generate_demos.py --aug-per-demo 4   # sim-verified human + augmented rollouts
python scripts/train_policy.py --encoding factorised --out checkpoints/aug_factorised.pt
python scripts/eval_policy.py --reference --ckpt checkpoints/aug_factorised.pt --n 60
python scripts/play_game.py --player human --video outputs/opera_game.mp4
python scripts/make_figures.py
pytest -q
```

The Menagerie Panda model is fetched automatically on first use. SmolVLA: open
`notebooks/smolvla_colab.ipynb` in Colab with a GPU runtime and run all cells.

## Repository layout

```
chessbot/
  env.py, scene.py, board.py, ik.py   MuJoCo chess environment
  quest_data.py                       loader for the Quest 3 / G1 teleop datasets
  calibration.py                      board pose from demonstrations
  retarget.py                         naive and object-centric retargeting
  augment.py, splits.py               board-symmetry augmentation, held-out splits
  policy.py                           compact task-conditioned policy
  player.py, game.py, oracle.py       full games: human-motion player, chess rules, scripted reference
scripts/                              one script per step above
notebooks/smolvla_colab.ipynb         SmolVLA fine-tuning and evaluation on Colab
calib/                                fitted board calibrations
artifacts/train_rollouts.jsonl.gz     the 3 195 sim-verified training rollouts
docs/                                 figures
```
