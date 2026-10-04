"""Locate (and on first use, fetch) the Franka Panda model from MuJoCo Menagerie."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_DIR = REPO_ROOT / "third_party" / "mujoco_menagerie"


def panda_dir() -> Path:
    """Return the directory holding Menagerie's ``franka_emika_panda`` model.

    Honours ``$MENAGERIE_PATH`` if set; otherwise sparse-clones just the Panda
    folder into ``third_party/`` the first time it is needed.
    """
    env = os.environ.get("MENAGERIE_PATH")
    root = Path(env) if env else MENAGERIE_DIR
    panda = root / "franka_emika_panda"
    if (panda / "panda.xml").exists():
        return panda
    if env:
        raise FileNotFoundError(f"$MENAGERIE_PATH={env} has no franka_emika_panda/panda.xml")
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", MENAGERIE_URL, str(root)],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", "franka_emika_panda"], check=True)
    return panda
