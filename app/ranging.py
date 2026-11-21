"""测点深度区间求解（与 Web 框架解耦，便于单元测试）。

业务含义
========
给定同一个物理测点在左、右 **去畸变** 图像中的像素坐标，以及一个统一的
像素容差 ``tol``，求该点在左相机坐标系下沿光轴（Z 轴）的深度范围：

    Z = { z | 存在 X=(x, y, z) 使得
              z > 0                       （左相机前方）
              (R X + T)_z > 0             （右相机前方）
              |u_l(X) - u_obs_l| <= tol,  |v_l(X) - v_obs_l| <= tol
              |u_r(X) - v_obs_r| <= tol,  |v_r(X) - v_obs_r| <= tol }

其中右相机投影使用已标定的 R, T：
    u_r = fx_r * (R X + T)_x / (R X + T)_z + cx_r，v_r 同理。

因为投影不等式在 z>0 时可两边乘 z 化成关于 X 的 **线性** 不等式，
可行域是三维半空间交（多面体），它在 Z 轴上的投影必为区间。于是：

* 区间为空            -> ``no_solution``（无可行位置）；
* 区间上端有限        -> ``bounded``，返回 z_min / z_max；
* 区间向上无限延伸    -> ``unbounded``，返回 z_min、z_max=None。

求解不引入 scipy：三维问题直接枚举半空间边界的交点（顶点）取极值，
再用“回收锥（recession cone）”判断是否向上无界。相机前方的约束
（z >= EPS、右相机 z_r >= EPS）保证多面体不含直线，因此可行域非空时
必有顶点，顶点枚举不会漏掉最小值。

注意：输入像素坐标按“已去畸变”解释，故此处只用内参 K，不使用畸变系数。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# 判定“位于相机前方”的数值下界（毫米）。仅用于排除 z=0 的退化边界。
EPS_DEPTH_MM = 1e-6

# 顶点可行性判定的相对容差（毫米，乘以 1+|X|）。
_FEASIBLE_REL_EPS = 1e-7

# 条件数超过该值的三个平面视为近似平行，跳过其交点。
_COND_LIMIT = 1e12


@dataclass
class DepthRange:
    """一次测点深度核验的几何结果。"""

    status: str                 # "bounded" | "unbounded" | "no_solution"
    z_min_mm: Optional[float]
    z_max_mm: Optional[float]
    tolerance_px: float
    half_space_count: int

    @property
    def bounded(self) -> bool:
        return self.status == "bounded"


def _pixel_constraints(fx: float, cx: float, u: float, tol: float,
                       c_basis: np.ndarray, c_offset: float,
                       z_basis: np.ndarray, z_offset: float
                       ) -> list[tuple[np.ndarray, float]]:
    """一个像素坐标轴的 ±tol 容差带，化为两条半空间。

    需要 c = c_basis·X + c_offset，z_cam = z_basis·X + z_offset。
    条件：|fx * c / z_cam + cx - u| <= tol，且 z_cam > 0。
    两边乘 z_cam 后：
        fx*c + (cx - u + tol) * z_cam >= 0
       -fx*c + (u + tol - cx) * z_cam >= 0
    即 a = ±fx*c_basis + (...)*z_basis，b 为常数项。
    """
    a_hi = fx * c_basis + (cx - u + tol) * z_basis
    b_hi = fx * c_offset + (cx - u + tol) * z_offset
    a_lo = -fx * c_basis + (u + tol - cx) * z_basis
    b_lo = -fx * c_offset + (u + tol - cx) * z_offset
    return [(a_hi, b_hi), (a_lo, b_lo)]


def _half_spaces(K_left: np.ndarray, K_right: np.ndarray,
                 R: np.ndarray, T: np.ndarray,
                 left_px: np.ndarray, right_px: np.ndarray,
                 tol: float) -> tuple[np.ndarray, np.ndarray]:
    """构造 A X + b >= 0 形式的全部半空间（已按 ||a|| 归一化）。"""
    fx_l, fy_l = float(K_left[0, 0]), float(K_left[1, 1])
    cx_l, cy_l = float(K_left[0, 2]), float(K_left[1, 2])
    fx_r, fy_r = float(K_right[0, 0]), float(K_right[1, 1])
    cx_r, cy_r = float(K_right[0, 2]), float(K_right[1, 2])
    if min(fx_l, fy_l, fx_r, fy_r) <= 0:
        raise ValueError("相机内参的焦距必须为正数")

    ul, vl = float(left_px[0]), float(left_px[1])
    ur, vr = float(right_px[0]), float(right_px[1])

    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3)
    r0, r1, r2 = R[0], R[1], R[2]
    t0, t1, t2 = float(T[0]), float(T[1]), float(T[2])

    rows: list[tuple[np.ndarray, float]] = []
    # 左相机：z > 0，以及 u/v 的 ±tol 容差
    rows.append((ez.copy(), -EPS_DEPTH_MM))
    rows += _pixel_constraints(fx_l, cx_l, ul, tol, ex, 0.0, ez, 0.0)
    rows += _pixel_constraints(fy_l, cy_l, vl, tol, ey, 0.0, ez, 0.0)

    # 右相机：X_r = R X + T，z_r > 0，以及 u/v 的 ±tol 容差
    rows.append((r2.copy(), t2 - EPS_DEPTH_MM))
    rows += _pixel_constraints(fx_r, cx_r, ur, tol, r0, t0, r2, t2)
    rows += _pixel_constraints(fy_r, cy_r, vr, tol, r1, t1, r2, t2)

    A = np.asarray([a for a, _ in rows], dtype=np.float64)
    b = np.asarray([c for _, c in rows], dtype=np.float64)
    norms = np.linalg.norm(A, axis=1)
    if np.any(norms <= 0):
        raise ValueError("内参或外参退化，无法构造投影约束")
    return A / norms[:, None], b / norms


def _vertices(A: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    """枚举可行多面体的全部顶点（三维，半空间交）。"""
    n = len(A)
    vertices: list[np.ndarray] = []
    for i, j, k in itertools.combinations(range(n), 3):
        M = A[[i, j, k]]
        if np.linalg.cond(M) > _COND_LIMIT:
            continue
        try:
            X = np.linalg.solve(M, -b[[i, j, k]])
        except np.linalg.LinAlgError:  # pragma: no cover - 条件数已过滤
            continue
        if not np.all(np.isfinite(X)):
            continue
        residual = A @ X + b
        eps = _FEASIBLE_REL_EPS * (1.0 + float(np.linalg.norm(X)))
        if np.all(residual >= -eps):
            vertices.append(X)
    return vertices


def _unbounded_above(A: np.ndarray) -> bool:
    """回收锥中是否存在 dz>0 的方向（即深度可无限增大）。"""
    n = len(A)
    target = np.array([0.0, 0.0, 1.0])
    for i, j in itertools.combinations(range(n), 2):
        M = np.vstack([A[i], A[j], target])
        if np.linalg.cond(M) > _COND_LIMIT:
            continue
        try:
            direction = np.linalg.solve(M, target)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        if not np.all(np.isfinite(direction)):
            continue
        if direction[2] <= 0:
            continue
        residual = A @ direction
        eps = 1e-9 * (1.0 + float(np.linalg.norm(direction)))
        if np.all(residual >= -eps):
            return True
    return False


def compute_depth_range(K_left: np.ndarray, K_right: np.ndarray,
                        R: np.ndarray, T: np.ndarray,
                        left_px: list[float] | tuple[float, float],
                        right_px: list[float] | tuple[float, float],
                        tolerance_px: float) -> DepthRange:
    """求测点沿左相机光轴的深度区间。

    ``R``、``T`` 为左相机到右相机的外参（X_right = R X_left + T）。
    输入像素坐标需为已去畸变坐标；``tolerance_px`` 非负，左右图共用。
    """
    tol = float(tolerance_px)
    if not math.isfinite(tol) or tol < 0:
        raise ValueError("像素容差必须是非负有限数")
    lp = np.asarray(left_px, dtype=np.float64).reshape(-1)
    rp = np.asarray(right_px, dtype=np.float64).reshape(-1)
    if lp.shape != (2,) or rp.shape != (2,):
        raise ValueError("左右像素坐标必须是 [u, v]")
    if not (np.all(np.isfinite(lp)) and np.all(np.isfinite(rp))):
        raise ValueError("像素坐标必须是有限数")

    K_left = np.asarray(K_left, dtype=np.float64)
    K_right = np.asarray(K_right, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if K_left.shape != (3, 3) or K_right.shape != (3, 3) or R.shape != (3, 3):
        raise ValueError("内参/旋转矩阵必须是 3x3")
    if T.size != 3:
        raise ValueError("平移向量必须是 3 个元素")

    A, b = _half_spaces(K_left, K_right, R, T, lp, rp, tol)
    vertices = _vertices(A, b)
    if not vertices:
        return DepthRange(
            status="no_solution", z_min_mm=None, z_max_mm=None,
            tolerance_px=tol, half_space_count=len(A),
        )

    z_values = np.asarray([v[2] for v in vertices], dtype=np.float64)
    z_min = float(np.min(z_values))
    if _unbounded_above(A):
        return DepthRange(
            status="unbounded", z_min_mm=z_min, z_max_mm=None,
            tolerance_px=tol, half_space_count=len(A),
        )
    return DepthRange(
        status="bounded", z_min_mm=z_min, z_max_mm=float(np.max(z_values)),
        tolerance_px=tol, half_space_count=len(A),
    )
