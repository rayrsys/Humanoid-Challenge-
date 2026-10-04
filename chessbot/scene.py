"""Programmatic MuJoCo scene: Menagerie Panda + table + chessboard + a pool of pieces.

The model is compiled once with a fixed *pool* of 34 piece bodies (a full set
plus a spare queen per colour for promotions). Pieces that are not on the
board are parked off to the side, so any FEN can be loaded by writing qpos.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .assets import panda_dir
from .board import BoardGeometry, PIECE_SHAPES

# Pool layout: symbol -> count, per colour.
POOL = {"p": 8, "n": 2, "b": 2, "r": 2, "q": 2, "k": 1}
PIECE_RGBA = {"w": (0.93, 0.89, 0.80, 1.0), "b": (0.16, 0.13, 0.12, 1.0)}
LIGHT_SQ = (0.92, 0.84, 0.68, 1.0)
DARK_SQ = (0.55, 0.37, 0.24, 1.0)
BORDER = (0.33, 0.21, 0.14, 1.0)

# Pinch point between the Panda fingertip pads, in the hand frame.
PINCH_OFFSET = (0.0, 0.0, 0.1034)
# Start pose: hand retracted towards the base so cameras see the whole board.
HOME_QPOS = np.array([0.0, -0.7648, 0.0, -2.8783, 0.0, 2.1132, 0.785398])
# Mid-workspace posture that the IK null-space is pulled towards.
POSTURE_QPOS = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785398])

_GEOM = {
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}


def pool_names() -> list[str]:
    """Body names of every piece in the pool, e.g. ``"w_n_1"``."""
    return [f"{c}_{s}_{i}" for c in "wb" for s, n in POOL.items() for i in range(n)]


def _look_at_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion for a MuJoCo camera at ``pos`` looking at ``target``."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight down/up
        right = np.array([0.0, -1.0, 0.0])
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, f)
    mat = np.stack([right, cam_up, -f], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q


def build_spec(geom: BoardGeometry | None = None) -> mujoco.MjSpec:
    geom = geom or BoardGeometry()
    spec = mujoco.MjSpec.from_file(str(panda_dir() / "panda.xml"))
    spec.modelname = "panda_chess"

    for key in list(spec.keys):  # keyframes would have the wrong qpos size
        spec.delete(key)

    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]

    # Fingertips: Menagerie models each pad as one flat 17x17 mm box plus four
    # small "bumps" that protrude 1.5 mm further. A cylinder pinched between
    # bump corners gets tilted contact normals that wedge it sideways out of
    # the grasp, so we keep only the flat pad (and the finger hull), and give
    # it torsional friction so a held piece cannot swing about the pinch axis.
    for name in ("left_finger", "right_finger"):
        for g in spec.body(name).geoms:
            if not (g.contype or g.conaffinity):
                continue
            if g.classname.name.startswith("fingertip_pad_collision_") and \
                    not g.classname.name.endswith("_1"):
                g.contype = g.conaffinity = 0
                continue
            g.friction = [1.5, 0.02, 0.0005]
            g.condim = 4

    # The real Panda controller compensates gravity; without it the PD
    # position servos sag by several millimetres at full reach.
    for name in [f"link{i}" for i in range(1, 8)] + ["hand", "left_finger", "right_finger"]:
        spec.body(name).gravcomp = 1.0

    # Menagerie's gripper is a 100 N/m spring (~1 N squeeze on a 2 cm piece);
    # stiffen it 4x so pieces do not slip during fast transport.
    grip = spec.actuator("actuator8")
    k = 4.0
    grip.gainprm[0] *= k
    grip.biasprm[1] *= k
    grip.biasprm[2] *= k

    hand = spec.body("hand")
    hand.add_site(name="pinch", pos=list(PINCH_OFFSET), size=[0.004, 0, 0], rgba=[1, 0, 0, 0.0])
    hand.add_camera(name="wrist", pos=[0.06, 0.0, 0.0], quat=[0, 0.7071068, 0.7071068, 0], fovy=75)

    for light in list(spec.lights):  # Menagerie's own light; we add our own below
        spec.delete(light)

    world = spec.worldbody
    world.add_light(name="key", pos=[0.4, -0.3, 1.6], dir=[0.1, 0.2, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    diffuse=[0.45, 0.45, 0.45], castshadow=1)
    world.add_light(name="fill", pos=[1.2, 0.6, 1.2], dir=[-0.6, -0.3, -0.7], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    diffuse=[0.15, 0.15, 0.15], castshadow=0)

    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.85, 0.88, 0.92], rgb2=[0.45, 0.5, 0.58], width=256, height=1536)
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.42, 0.44, 0.47], rgb2=[0.38, 0.40, 0.43], width=256, height=256)
    floor_mat = spec.add_material(name="floor")
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
    floor_mat.texrepeat = [8, 8]

    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05],
                   pos=[0, 0, -0.76], material="floor", contype=0, conaffinity=0)
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.65, 0.38],
                   pos=[0.4, 0.0, -0.38], rgba=[0.42, 0.40, 0.38, 1], friction=[0.8, 0.01, 0.001])

    # Board: one collision slab plus 64 visual-only squares on top.
    o = geom.to_world(np.zeros(3))
    half = 4 * geom.square + geom.border
    world.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[half, half, (o[2]) / 2], pos=[o[0], o[1], o[2] / 2],
                   rgba=list(BORDER), friction=[0.9, 0.01, 0.001])
    for sq in range(64):
        f, r = sq % 8, sq // 8
        xy = geom.to_world(np.array([(r - 3.5) * geom.square, (3.5 - f) * geom.square, 0.0]))
        dark = (f + r) % 2 == 0  # a1 is dark
        world.add_geom(name=f"sq_{sq}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[geom.square / 2, geom.square / 2, 0.0005],
                       pos=[xy[0], xy[1], o[2] - 0.0003], rgba=list(DARK_SQ if dark else LIGHT_SQ),
                       contype=0, conaffinity=0)

    # Piece pool, initially parked; positions are set at reset.
    k = geom.scale
    for idx, name in enumerate(pool_names()):
        color, sym, _ = name.split("_")
        shape = PIECE_SHAPES[sym]
        body = world.add_body(name=name, pos=[0.0, 1.0 + 0.05 * idx, 0.1])
        body.add_freejoint(name=f"{name}_free")
        for j, (gtype, size, zc, xoff) in enumerate(shape.parts):
            sz = [v * k for v in size] + [0.0] * (3 - len(size))
            # Knights face the opponent: White's look +x, Black's -x.
            x = xoff * k * (1 if color == "w" else -1)
            body.add_geom(name=f"{name}_g{j}", type=_GEOM[gtype], size=sz, pos=[x, 0.0, zc * k],
                          rgba=list(PIECE_RGBA[color]), density=1500, condim=4,
                          friction=[1.2, 0.01, 0.0005], solref=[0.004, 1])
        body.add_site(name=f"{name}_grasp", pos=[0, 0, shape.grasp_z * k], size=[0.002, 0, 0],
                      rgba=[0, 0, 0, 0])
        # Grasp stabiliser (inactive until the env detects a two-finger squeeze).
        spec.add_equality(name=f"{name}_hold", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1="hand", name2=name, active=False)

    # Cameras. "front" mirrors LIBERO's agentview: across the table, looking back at the robot.
    centre = geom.to_world(np.zeros(3))
    for name, pos, target, fovy in (
        ("front", centre + [0.62, 0.0, 0.48], centre + [-0.06, 0.0, 0.0], 48),
        ("side", centre + [0.05, -0.85, 0.55], centre + [0.0, 0.0, 0.02], 45),
        ("top", centre + [0.0, 0.0, 0.9], centre, 32),
        ("player", centre + [-0.62, -0.32, 0.58], centre + [0.05, 0.0, 0.0], 50),
    ):
        world.add_camera(name=name, pos=list(pos), quat=list(_look_at_quat(pos, target)), fovy=fovy)
    return spec


def build_model(geom: BoardGeometry | None = None) -> mujoco.MjModel:
    return build_spec(geom).compile()
