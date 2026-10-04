"""Train / evaluation split used by every experiment.

* ``HELD_OUT_SQUARES`` never appear in training, as source or destination;
  tasks touching them measure generalisation to unseen squares.
* ``held_out_pair`` hashes (piece, src, dst) and holds out ~10% of the
  remaining combinations; both squares are seen in training, the pairing is not.
"""

from __future__ import annotations

import hashlib

HELD_OUT_SQUARES = {"b6", "c3", "d7", "e2", "f5", "g4"}


def held_out_pair(piece: str, src: str, dst: str, frac: float = 0.10) -> bool:
    h = int(hashlib.sha1(f"{piece}:{src}:{dst}".encode()).hexdigest(), 16)
    return (h % 1000) < frac * 1000


def split_of(piece: str, src: str, dst: str) -> str:
    if src in HELD_OUT_SQUARES or dst in HELD_OUT_SQUARES:
        return "unseen_square"
    if held_out_pair(piece, src, dst):
        return "unseen_pair"
    return "train"
