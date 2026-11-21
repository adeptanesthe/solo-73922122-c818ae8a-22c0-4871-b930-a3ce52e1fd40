"""测点深度区间几何测试：正确性、退化情形与三类状态。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.ranging import compute_depth_range

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_calibration.json"


@pytest.fixture(scope="module")
def sample():
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return (
        data,
        np.asarray(data["left_camera"]["camera_matrix"], np.float64),
        np.asarray(data["right_camera"]["camera_matrix"], np.float64),
        np.asarray(data["_ground_truth"]["R_left_to_right"], np.float64),
        np.asarray(data["_ground_truth"]["T_left_to_right"], np.float64),
    )


def _project(K: np.ndarray, X: np.ndarray) -> list[float]:
    q = K @ np.asarray(X, np.float64).reshape(3)
    return [float(q[0] / q[2]), float(q[1] / q[2])]


def _pixels(Kl, Kr, R, T, X):
    Xr = R @ np.asarray(X, np.float64) + T
    return _project(Kl, X), _project(Kr, Xr)


def test_zero_tolerance_recovers_exact_depth(sample):
    _, Kl, Kr, R, T = sample
    X = np.array([30.0, -20.0, 920.0])
    lp, rp = _pixels(Kl, Kr, R, T, X)
    out = compute_depth_range(Kl, Kr, R, T, lp, rp, 0.0)
    assert out.status == "bounded"
    assert out.z_min_mm == pytest.approx(920.0, abs=1e-6)
    assert out.z_max_mm == pytest.approx(920.0, abs=1e-6)


def test_tolerance_interval_contains_truth_and_grows(sample):
    _, Kl, Kr, R, T = sample
    X = np.array([100.0, 50.0, 1000.0])
    lp, rp = _pixels(Kl, Kr, R, T, X)
    prev_width = -1.0
    for tol in (0.0, 0.5, 1.0, 2.0):
        out = compute_depth_range(Kl, Kr, R, T, lp, rp, tol)
        assert out.status == "bounded"
        assert out.z_min_mm <= 1000.0 <= out.z_max_mm
        width = out.z_max_mm - out.z_min_mm
        assert width >= prev_width
        prev_width = width
    assert prev_width > 0


def test_sample_probe_matches_ground_truth():
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    probe = data["depth_probe"]
    Kl = np.asarray(data["left_camera"]["camera_matrix"], np.float64)
    Kr = np.asarray(data["right_camera"]["camera_matrix"], np.float64)
    R = np.asarray(data["_ground_truth"]["R_left_to_right"], np.float64)
    T = np.asarray(data["_ground_truth"]["T_left_to_right"], np.float64)
    exact = compute_depth_range(
        Kl, Kr, R, T, probe["left_px"], probe["right_px"], 0.0
    )
    truth_z = data["_ground_truth"]["probe_point_left_mm"][2]
    assert exact.status == "bounded"
    assert exact.z_min_mm == pytest.approx(truth_z, abs=1e-3)
    wide = compute_depth_range(
        Kl, Kr, R, T, probe["left_px"], probe["right_px"], probe["tolerance_px"]
    )
    assert wide.status == "bounded"
    assert wide.z_min_mm < truth_z < wide.z_max_mm


def test_mismatched_pixels_have_no_solution(sample):
    _, Kl, Kr, R, T = sample
    X = np.array([40.0, -25.0, 900.0])
    lp, rp = _pixels(Kl, Kr, R, T, X)
    # 把右图像素横向移动 400 px，任何容差 1 px 的点都不满足对极约束
    out = compute_depth_range(Kl, Kr, R, T, lp, [rp[0] - 400.0, rp[1]], 1.0)
    assert out.status == "no_solution"
    assert out.z_min_mm is None and out.z_max_mm is None


def test_large_tolerance_reports_unbounded(sample):
    _, Kl, Kr, R, T = sample
    X = np.array([40.0, -25.0, 900.0])
    lp, rp = _pixels(Kl, Kr, R, T, X)
    out = compute_depth_range(Kl, Kr, R, T, lp, rp, 80.0)
    assert out.status == "unbounded"
    assert out.z_min_mm is not None and out.z_min_mm > 0
    assert out.z_max_mm is None


def test_depth_behind_cameras_rejected(sample):
    """相机前方约束生效：不存在 z<=0 的可行点。"""
    _, Kl, Kr, R, T = sample
    lp = [650.0, 480.0]
    out = compute_depth_range(Kl, Kr, R, T, lp, lp, 1.0)
    if out.status != "no_solution":
        assert out.z_min_mm > 0
        # 若给出区间，则区间内任一深度都满足左右投影容差
        depth = 0.5 * (out.z_min_mm + out.z_max_mm)
        assert depth > 0


def test_vertex_z_never_negative_on_random_inputs(sample):
    """随机扰动像素下，状态自洽且所有给出的深度均为正。"""
    _, Kl, Kr, R, T = sample
    rng = np.random.default_rng(11)
    for _ in range(60):
        lp = [rng.uniform(0, 1280), rng.uniform(0, 960)]
        rp = [rng.uniform(0, 1280), rng.uniform(0, 960)]
        tol = float(rng.choice([0.0, 0.5, 1.0, 5.0, 30.0]))
        out = compute_depth_range(Kl, Kr, R, T, lp, rp, tol)
        if out.status == "no_solution":
            assert out.z_min_mm is None and out.z_max_mm is None
            continue
        assert out.z_min_mm > 0
        if out.status == "bounded":
            assert out.z_max_mm >= out.z_min_mm
        else:
            assert out.z_max_mm is None


def test_invalid_inputs_rejected(sample):
    _, Kl, Kr, R, T = sample
    with pytest.raises(ValueError):
        compute_depth_range(Kl, Kr, R, T, [0, 0], [0, 0], -1.0)
    with pytest.raises(ValueError):
        compute_depth_range(Kl, Kr, R, T, [0, 0], [0, 0], float("nan"))
    with pytest.raises(ValueError):
        compute_depth_range(Kl, Kr, R, T, [0, 0, 0], [0, 0], 1.0)
    with pytest.raises(ValueError):
        compute_depth_range(Kl[:2], Kr, R, T, [0, 0], [0, 0], 1.0)


def test_zero_focal_length_rejected(sample):
    _, Kl, Kr, R, T = sample
    bad = Kl.copy()
    bad[0, 0] = 0.0
    with pytest.raises(ValueError):
        compute_depth_range(bad, Kr, R, T, [640, 480], [640, 480], 1.0)


def test_synthetic_parallel_cameras_exact_depth():
    """纯平移（R=I）的一对平行相机，零容差应精确还原深度。"""
    K = np.array([[800.0, 0, 640], [0, 800, 480], [0, 0, 1]], np.float64)
    R = np.eye(3)
    T = np.array([-120.0, 0.0, 0.0])
    X = np.array([80.0, 60.0, 1500.0])
    lp = _project(K, X)
    rp = _project(K, X + T)
    out = compute_depth_range(K, K, R, T, lp, rp, 0.0)
    assert out.status == "bounded"
    assert out.z_min_mm == pytest.approx(1500.0, abs=1e-6)
    assert out.z_max_mm == pytest.approx(1500.0, abs=1e-6)


def test_cross_checked_against_scipy_when_available():
    """若环境装有 scipy，用 linprog 独立校验区间端点（不进入运行依赖）。"""
    scipy = pytest.importorskip("scipy.optimize")
    from app.ranging import _half_spaces

    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    Kl = np.asarray(data["left_camera"]["camera_matrix"], np.float64)
    Kr = np.asarray(data["right_camera"]["camera_matrix"], np.float64)
    R = np.asarray(data["_ground_truth"]["R_left_to_right"], np.float64)
    T = np.asarray(data["_ground_truth"]["T_left_to_right"], np.float64)
    rng = np.random.default_rng(3)
    for _ in range(80):
        X = np.array([rng.uniform(-400, 400), rng.uniform(-300, 300),
                      rng.uniform(300, 3000)])
        lp, rp = _pixels(Kl, Kr, R, T, X)
        if _ % 4 == 0:
            rp = [rp[0] + rng.uniform(-300, 300), rp[1] + rng.uniform(-300, 300)]
        tol = float(rng.choice([0.0, 0.5, 1.0, 5.0, 40.0]))
        out = compute_depth_range(Kl, Kr, R, T, lp, rp, tol)
        A, b = _half_spaces(Kl, Kr, R, T, lp, rp, tol)
        lo = scipy.linprog([0, 0, 1.0], A_ub=-A, b_ub=b,
                           bounds=[(None, None)] * 3, method="highs")
        hi = scipy.linprog([0, 0, -1.0], A_ub=-A, b_ub=b,
                           bounds=[(None, None)] * 3, method="highs")
        if out.status == "no_solution":
            assert not lo.success
        elif out.status == "unbounded":
            assert lo.success and not hi.success
            assert out.z_min_mm == pytest.approx(lo.fun, rel=1e-9, abs=1e-6)
        else:
            assert lo.success and hi.success
            assert out.z_min_mm == pytest.approx(lo.fun, rel=1e-9, abs=1e-6)
            assert out.z_max_mm == pytest.approx(-hi.fun, rel=1e-9, abs=1e-6)
