"""Play a whole chess game on the Panda.

    MUJOCO_GL=egl python scripts/play_game.py --player human --video outputs/opera_game.mp4
    MUJOCO_GL=egl python scripts/play_game.py --player policy --ckpt checkpoints/aug_fact.pt

Players:
  human   every move is executed with a retrieved, re-anchored teleop motion (chessbot/player.py)
  policy  the trained task-conditioned policy, prompted with "Move the <colour> <piece> from x to y."
  oracle  the scripted reference controller

If a physical move fails, the piece is snapped back onto its square so the
game can go on; every such intervention is counted and reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no install needed

import argparse
import io
import json

import chess
import chess.pgn
import mujoco
import numpy as np

OPERA_GAME = """
[Event "Paris"] [White "Paul Morphy"] [Black "Duke Karl / Count Isouard"] [Date "1858"]
1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7
8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7
14. Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0
"""


def load_trajectories(data=None):
    from chessbot.calibration import BoardCalibration
    from chessbot.quest_data import load_dataset
    from chessbot.retarget import extract

    out = []
    for arm, name in (("right", "fb32-v03-all5-right"), ("left", "fb32-v03-all5-left")):
        cal = BoardCalibration.load(f"calib/board_{arm}.json")
        out += [extract(e, cal) for e in load_dataset(name, data) if e.labelled]
    return out


def snap(env, sq: int):
    """Intervention: put the piece on ``sq`` back upright at the square centre."""
    name = env.occupant[sq]
    env._place(name, env.geom.square_xy(sq))
    mujoco.mj_forward(env.model, env.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", choices=["human", "policy", "oracle"], default="human")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pgn", default=None, help="PGN file (default: Morphy's Opera Game)")
    ap.add_argument("--plies", type=int, default=0, help="stop after N plies (0 = whole game)")
    ap.add_argument("--video", default=None)
    ap.add_argument("--camera", default="player")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    from chessbot.env import ChessEnv
    from chessbot.game import execute_move

    game = chess.pgn.read_game(io.StringIO(open(args.pgn).read() if args.pgn else OPERA_GAME))
    moves = list(game.mainline_moves())[: args.plies or None]
    env = ChessEnv(cameras=())
    env.reset(seed=args.seed)
    frames = []

    def record(_env=None, *_):
        if args.video and env.t % 2 == 0:
            frames.append(env.render(args.camera, size=(480, 640)))

    if args.player == "human":
        from chessbot.player import HumanMotionPlayer
        player = HumanMotionPlayer(load_trajectories(args.data), seed=args.seed)
        mover = lambda env, a, b: player.mover(env, a, b, on_step=record)
    elif args.player == "policy":
        import torch
        from chessbot.policy import PolicyRunner, TaskPolicy
        c = torch.load(args.ckpt, weights_only=False)
        model = TaskPolicy(c["encoding"], c["horizon"])
        model.load_state_dict(c["state"])
        runner = PolicyRunner(model)

        def mover(env, a, b):
            name = env.occupant[a]
            colour = "white" if name[0] == "w" else "black"
            piece = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}[name.split("_")[1]]
            runner.run(env, f"Move the {colour} {piece} from {chess.square_name(a)} to {chess.square_name(b)}.",
                       on_step=record)
    else:
        from chessbot.oracle import oracle_mover
        mover = lambda env, a, b: oracle_mover(env, a, b, on_step=lambda _o: record())

    log, interventions = [], 0
    for ply, mv in enumerate(moves):
        san = env.board.san(mv)
        out = execute_move(env, mv, mover)
        for uci, ok, err, *_ in out.primitives:
            if not ok:
                interventions += 1
                snap(env, chess.parse_square(uci[2:]))
        # Pieces nudged off their squares by this move are put back too.
        for sq, name in list(env.occupant.items()):
            p, R = env.piece_pose(name)
            if np.linalg.norm(p[:2] - env.geom.square_xy(sq)) > 0.3 * env.geom.square or R[2, 2] < 0.95:
                interventions += 1
                snap(env, sq)
        log.append(dict(ply=ply + 1, san=san, primitives=out.primitives))
        status = "ok" if out.success else "FIXED (" + "; ".join(p[3] for p in out.primitives if not p[1]) + ")"
        print(f"{ply // 2 + 1}{'.' if ply % 2 == 0 else '...'} {san:8s} {status}", flush=True)
    n_prim = sum(len(x["primitives"]) for x in log)
    n_ok = sum(p[1] for x in log for p in x["primitives"])
    print(json.dumps(dict(player=args.player, plies=len(moves), primitives=n_prim,
                          primitive_success=n_ok / max(n_prim, 1), interventions=interventions), indent=2))
    if args.player == "human":
        print("motions used (move, source demo, mirrored):", player.used[:6], "...")
    if args.video and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=25)
        print("wrote", args.video)


if __name__ == "__main__":
    main()
