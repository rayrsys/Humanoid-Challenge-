import chess
import numpy as np
import pytest

from chessbot.board import PIECE_SHAPES, HAND_PALM_Z, BoardGeometry


def test_square_roundtrip():
    g = BoardGeometry()
    for sq in chess.SQUARES:
        assert g.xy_to_square(g.square_xy(sq)) == sq


def test_board_orientation():
    g = BoardGeometry()
    # rank 1 is nearest the robot (-x), the a-file is on White's left (+y)
    assert g.square_xy("a1")[0] < g.square_xy("a8")[0]
    assert g.square_xy("a1")[1] > g.square_xy("h1")[1]


def test_pieces_clear_the_palm_and_fit_the_pads():
    for sym, s in PIECE_SHAPES.items():
        assert s.height - s.grasp_z < HAND_PALM_Z - 0.003, sym
        assert s.grasp_z - 0.008 > 0.012, sym  # pads stay above the base collar


def test_yaw_roundtrip():
    from chessbot.ik import grasp_rotation, yaw_of
    for yaw in np.linspace(-3, 3, 13):
        assert np.isclose(np.angle(np.exp(1j * (yaw_of(grasp_rotation(yaw)) - yaw))), 0, atol=1e-9)


def test_quaternion_matrix_is_rotation():
    from chessbot.calibration import quat_wxyz_to_mat
    q = np.random.default_rng(0).normal(size=(20, 4))
    R = quat_wxyz_to_mat(q)
    assert np.allclose(R @ np.swapaxes(R, 1, 2), np.eye(3), atol=1e-9)
    assert np.allclose(np.linalg.det(R), 1)


def test_augment_preserves_move_vector():
    from chessbot.augment import _shift
    for src, dst in [("g1", "f3"), ("c1", "h6"), ("a1", "a8")]:
        for df, dr, m in [(1, 2, False), (-2, 0, True), (0, -1, True)]:
            s2, d2 = _shift(src, df, dr, m), _shift(dst, df, dr, m)
            if s2 is None or d2 is None:
                continue
            v = np.subtract(divmod(chess.parse_square(dst), 8), divmod(chess.parse_square(src), 8))
            v2 = np.subtract(divmod(chess.parse_square(d2), 8), divmod(chess.parse_square(s2), 8))
            assert v2[0] == v[0] and abs(v2[1]) == abs(v[1])  # rank delta kept; file delta kept or mirrored


@pytest.mark.slow
def test_ik_reaches_every_square():
    from chessbot.env import ChessEnv
    from chessbot.ik import grasp_rotation
    from chessbot.scene import HOME_QPOS
    env = ChessEnv(cameras=())
    for sq in chess.SQUARES:
        q, err = env.ik.solve(HOME_QPOS, env.geom.square_world(sq, 0.03), grasp_rotation(0.0), iters=300)
        assert err < 1e-3, chess.square_name(sq)
