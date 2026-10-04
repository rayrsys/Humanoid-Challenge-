"""Figures for the README (static PNGs on a light surface).

    python scripts/make_figures.py            # writes docs/fig_*.png from outputs/*.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no install needed

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order (light)
DOCS = Path("docs")


def _style(ax, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, length=0, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=22 if subtitle else 10, fontweight="bold")
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=INK2)


def _fig(w=8.0, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def retargeting():
    res = {m: json.loads(Path(f"outputs/replay_{m}.json").read_text()) for m in ("naive", "object_centric")}
    pieces = ["knight", "bishop", "rook", "queen", "king"]
    groups = pieces + ["all"]
    fig, ax = _fig()
    w = 0.34
    x = np.arange(len(groups))
    for i, (m, label) in enumerate((("naive", "Naive (copy the hand)"), ("object_centric", "Object-centric (ours)"))):
        vals = [np.mean([r["success"] for r in res[m] if g == "all" or r["piece"] == g]) * 100 for g in groups]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=SERIES[1 - i], edgecolor=SURFACE, linewidth=2, label=label)
        b = bars[-1]  # direct label on the headline group only
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{vals[-1]:.0f}%", ha="center",
                va="bottom", fontsize=10, color=INK)
    n = len(res["naive"])
    ax.set_xticks(x, [g.capitalize() if g != "all" else "All moves" for g in groups])
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    _style(ax, "Retargeting G1 teleop demos onto a Panda",
           f"Open-loop replay success in MuJoCo, all {n} labelled Quest 3 episodes")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=10, labelcolor=INK2, ncol=2)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_retargeting.png", facecolor=SURFACE)
    plt.close(fig)


def policies(eval_path="outputs/eval.json"):
    table = json.loads(Path(eval_path).read_text())
    splits = [("train", "Seen tasks"), ("unseen_pair", "Unseen square pairs"), ("unseen_square", "Unseen squares")]
    models = [("aug_factorised", "Augmented · factorised squares"), ("aug_flat", "Augmented · flat squares"),
              ("human_factorised", "Human only · factorised"), ("human_flat", "Human only · flat")]
    models = [(k, l) for k, l in models if k in table]
    ref = table.get("human replay (reference)")
    fig, ax = _fig(9.0, 4.4)
    x = np.arange(len(splits))
    w = 0.8 / len(models)
    for i, (k, label) in enumerate(models):
        vals = [table[k][s] * 100 for s, _ in splits]
        bars = ax.bar(x - 0.4 + (i + 0.5) * w, vals, min(w, 0.2), color=SERIES[i], edgecolor=SURFACE, linewidth=2,
                      label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5,
                    color=INK2)
    if ref:
        for j, (s, _) in enumerate(splits):
            ax.plot([j - 0.42, j + 0.42], [ref[s] * 100] * 2, color=INK, linewidth=1.2, linestyle=(0, (4, 2)))
        ax.plot([], [], color=INK, linewidth=1.2, linestyle=(0, (4, 2)), label="Human replay (reference)")
    ax.set_xticks(x, [l for _, l in splits])
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    _style(ax, "Closed-loop success of policies trained on retargeted demos",
           "Same seeded task set per split; unseen squares never appear in training")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_policies.png", facecolor=SURFACE)
    plt.close(fig)


def calibration():
    from chessbot.calibration import calibrate
    from chessbot.quest_data import load_dataset

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, arm in zip(axes, ("right", "left")):
        cal, diag = calibrate(load_dataset(f"fb32-v03-all5-{arm}"), iters=4)
        R = np.array([r[1] for r in diag["rows"]])
        from chessbot.calibration import quat_wxyz_to_mat
        P = np.array([r[0] for r in diag["rows"]]) + np.einsum("nij,j->ni", quat_wxyz_to_mat(R), cal.offset)
        b = cal.to_board(P)
        for f in range(8):
            for k in range(8):
                ax.add_patch(plt.Rectangle((f - 0.5, k - 0.5), 1, 1, color="#ecebe6" if (f + k) % 2 else "#f7f6f3",
                                           zorder=0))
        kinds = np.array([r[5] for r in diag["rows"]])
        for i, (kind, label) in enumerate((("grasp", "grasp"), ("release", "release"))):
            m = kinds == kind
            ax.scatter(b[m, 0], b[m, 1], s=10, color=SERIES[i], edgecolors=SURFACE, linewidths=0.6, label=label,
                       alpha=0.9, zorder=2)
        ax.set_xlim(-0.6, 7.6)
        ax.set_ylim(-0.6, 7.6)
        ax.set_aspect("equal")
        ax.set_xticks(range(8), list("abcdefgh"))
        ax.set_yticks(range(8), [str(i) for i in range(1, 9)])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK2, length=0)
        res = diag["residual"] * 1000
        ax.set_title(f"{arm.capitalize()} arm: square {cal.square * 1000:.1f} mm, median error "
                     f"{np.median(res[diag['keep']]):.1f} mm", fontsize=10, color=INK, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, fontsize=10, labelcolor=INK2,
               markerscale=1.6)
    fig.suptitle("Board recovered from the demonstrations alone: fingertip point at every grasp and release",
                 x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(DOCS / "fig_calibration.png", facecolor=SURFACE)
    plt.close(fig)


def side_by_side(episodes=((("right", 0)), ("right", 105)), out="fig_teleop_vs_panda.png"):
    """G1 teleop head-camera frames next to the Panda executing the retargeted motion."""
    import subprocess

    import chess
    import pandas as pd

    from chessbot.calibration import BoardCalibration
    from chessbot.env import ChessEnv
    from chessbot.quest_data import load_dataset
    from chessbot.retarget import extract, retarget_object_centric
    from scripts.replay import obstacles

    env = ChessEnv(cameras=())
    rows = []
    for arm, ep in episodes:
        root = f"data/fb32-v03-all5-{arm}"
        meta = pd.read_parquet(f"{root}/meta/episodes/chunk-000/file-000.parquet")
        row = meta[meta.episode_index == ep].iloc[0]
        key = "videos/observation.images.head"
        video = f"{root}/videos/observation.images.head/chunk-{int(row[key + '/chunk_index']):03d}/" \
                f"file-{int(row[key + '/file_index']):03d}.mp4"
        e = {x.index: x for x in load_dataset(f"fb32-v03-all5-{arm}")}[ep]
        ht = extract(e, BoardCalibration.load(f"calib/board_{arm}.json"))
        env.reset(fen=e.fen)
        acts, src_idx = retarget_object_centric(ht, env.geom, trim=True, obstacles=obstacles(env, e.src),
                                                return_source_index=True)
        tg, tr = ht.t_grasp, ht.t_release
        keys = [tg, (tg + tr) // 2, tr]
        # Panda step where each key frame is reached (first step retargeted from it or later).
        want = {k: int(np.argmax(src_idx >= k)) for k in keys}
        panda = {}
        for i, a in enumerate(acts):
            env.step(a)
            for k, s in want.items():
                if s == i:
                    panda[k] = env.render("front", size=(240, 320))
        g1 = []
        for k in keys:
            raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(row[key + "/from_timestamp"] + k / 50), "-i", video,
                                  "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
            g1.append(np.frombuffer(raw, np.uint8).reshape(240, 320, 3))
        ok = env.check_move(chess.Move.from_uci(e.src + e.dst)).success
        rows.append((e.task, g1, [panda[k] for k in keys], ok))

    # Compose one canvas (no per-axes layout): per episode a G1 row over a Panda row.
    gap, head = 6, 34
    w, h = 320, 240
    W = 3 * w + 2 * gap
    H = len(rows) * (head + 2 * h + gap) + 30
    canvas = np.full((H, W, 3), 252, np.uint8)
    y = 30
    labels = []
    for task, g1, pd_frames, ok in rows:
        labels.append((y + 22, f"{task}   Panda: {'success' if ok else 'failure'}"))
        y += head
        for r, frames in enumerate((g1, pd_frames)):
            for c, img in enumerate(frames):
                canvas[y + r * (h + gap) // 1: y + r * (h + gap) + h, c * (w + gap): c * (w + gap) + w] = img
        y += 2 * h + gap
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(canvas)
    ax.axis("off")
    for c, name in enumerate(("grasp", "carry", "release")):
        ax.text(c * (w + gap) + w / 2, 20, name, ha="center", fontsize=11, color=INK2)
    for yy, text in labels:
        ax.text(4, yy, text, fontsize=11, color=INK, fontweight="bold")
    for i in range(len(rows)):
        y0 = 30 + i * (head + 2 * h + gap) + head
        box = dict(facecolor=INK, alpha=0.65, edgecolor="none", pad=2)
        ax.text(6, y0 + 16, "G1 teleop (Quest 3)", fontsize=9, color="white", fontweight="bold", bbox=box)
        ax.text(6, y0 + h + gap + 16, "Panda (retargeted)", fontsize=9, color="white", fontweight="bold", bbox=box)
    fig.savefig(DOCS / out, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    DOCS.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["retargeting", "calibration", "policies", "side_by_side"]
    for name in which:
        try:
            globals()[name]()
            print("done:", name)
        except FileNotFoundError as e:
            print(f"skipped {name}: {e}")
