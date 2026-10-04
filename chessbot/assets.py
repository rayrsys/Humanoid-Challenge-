"""Locate (and on first use, fetch) the Franka Panda model from MuJoCo Menagerie."""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_DIR = REPO_ROOT / "third_party" / "mujoco_menagerie"


def panda_dir() -> Path:
    """Return the directory holding Menagerie's ``franka_emika_panda`` model.

    Honours ``$MENAGERIE_PATH`` if set; otherwise sparse-clones just the Panda
    folder into ``third_party/`` the first time it is needed. Safe to call from
    many processes at once: one clones (into a temporary folder that is then
    renamed into place) while the others wait on a file lock.
    """
    env = os.environ.get("MENAGERIE_PATH")
    root = Path(env) if env else MENAGERIE_DIR
    panda = root / "franka_emika_panda"
    if (panda / "panda.xml").exists():
        return panda
    if env:
        raise FileNotFoundError(f"$MENAGERIE_PATH={env} has no franka_emika_panda/panda.xml")
    root.parent.mkdir(parents=True, exist_ok=True)
    with open(root.parent / ".menagerie.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if (panda / "panda.xml").exists():  # another process fetched it while we waited
                return panda
            if root.exists():  # leftover from an interrupted fetch
                shutil.rmtree(root)
            tmp = Path(tempfile.mkdtemp(prefix="menagerie-", dir=root.parent))
            subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                            MENAGERIE_URL, str(tmp / "repo")], check=True)
            subprocess.run(["git", "-C", str(tmp / "repo"), "sparse-checkout", "set", "franka_emika_panda"],
                           check=True)
            (tmp / "repo").rename(root)
            shutil.rmtree(tmp, ignore_errors=True)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return panda
