"""FastAPI 网页后端：上传标定 JSON、执行外参标定、留存上次成功结果。"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .calibration import (
    BoardSpec,
    CalibrationOutcome,
    CameraParams,
    CandidateResult,
    PairResult,
    run_calibration,
)
from .ranging import compute_depth_range

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="双目相机外参标定", version="1.0.0")

# ---- 请求/响应模型 ----------------------------------------------------------


class IntrinsicInput(BaseModel):
    camera_matrix: list[list[float]] = Field(..., description="3x3 内参矩阵 K")
    dist_coeffs: list[float] = Field(..., description="畸变系数（去畸变数据可全 0）")


class BoardInput(BaseModel):
    rows: int = Field(..., gt=1, description="棋盘格内角点行数")
    cols: int = Field(..., gt=1, description="棋盘格内角点列数")
    square_size: float = Field(..., gt=0, description="格距（毫米）")


class ImagePairInput(BaseModel):
    pair_id: str
    left_pts: list[list[float]] = Field(..., description="左图角点 [[x,y],...]")
    right_pts: list[list[float]] = Field(..., description="右图角点 [[x,y],...]")
    image_size: Optional[list[int]] = Field(
        None, description="[width, height]，缺省时由角点范围推断"
    )


class CalibrationPayload(BaseModel):
    left_camera: IntrinsicInput
    right_camera: IntrinsicInput
    board: BoardInput
    image_pairs: list[ImagePairInput] = Field(..., min_length=3, max_length=6)
    selected_pair_ids: list[str] = Field(..., min_length=3, max_length=6)
    pixel_threshold: float = Field(..., gt=0)


class CalibrateRequest(BaseModel):
    data: CalibrationPayload
    session_id: Optional[str] = None


class RangeRequest(BaseModel):
    """测点深度核验请求。seq 由前端单调递增，用于拒绝迟到的旧结果。"""

    session_id: str
    calib_token: str
    request_seq: int = Field(..., ge=1)
    left_px: list[float] = Field(..., min_length=2, max_length=2)
    right_px: list[float] = Field(..., min_length=2, max_length=2)
    tolerance_px: float = Field(..., ge=0, allow_inf_nan=False)
    min_depth_mm: float = Field(..., ge=0, allow_inf_nan=False)
    max_depth_mm: float = Field(..., gt=0, allow_inf_nan=False)


# ---- 进程内状态 -------------------------------------------------------------


class Store:
    """留存本批上传数据、上次成功标定结果与最新测距结果。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session_id: Optional[str] = None
        self.raw: Optional[dict] = None
        self.last_success: Optional[dict] = None
        self.calib_token: Optional[str] = None
        self.range_result: Optional[dict] = None
        self.range_seq: int = 0

    def reset_with(self, raw: dict) -> None:
        with self.lock:
            # 新一批数据：清掉上一批的成功结果与测距结果
            self.session_id = uuid.uuid4().hex[:12]
            self.raw = raw
            self.last_success = None
            self.calib_token = None
            self.range_result = None
            self.range_seq = 0

    def accept_calibration(self, result: dict, intrinsics: dict,
                           session_id: Optional[str] = None
                           ) -> Optional[str]:
        """记录一次成功标定；旧测距结果随之作废。

        带 ``session_id`` 时，若批次已切换（标定请求迟到）则拒收并返回 None。
        """
        with self.lock:
            if session_id is not None and session_id != self.session_id:
                return None
            token = uuid.uuid4().hex[:12]
            self.calib_token = token
            self.last_success = {
                "session_id": self.session_id,
                "calib_token": token,
                "result": result,
                "intrinsics": intrinsics,
            }
            self.range_result = None
            self.range_seq = 0
            return token


STORE = Store()


# ---- 序列化 -----------------------------------------------------------------


def _pair_to_dict(pr: PairResult) -> dict[str, Any]:
    return {
        "pair_id": pr.pair_id,
        "order_name": pr.order_name,
        "per_pair_rms": round(pr.per_pair_rms, 6),
        "left_rms": round(pr.left_rms, 6),
        "right_rms": round(pr.right_rms, 6),
        "point_count": pr.point_count,
        "passed": pr.passed,
        "independent_rms": None if pr.independent_rms is None else round(pr.independent_rms, 6),
        "rotation_delta_deg": None if pr.rot_deg is None else round(pr.rot_deg, 4),
        "translation_angle_deg": None if pr.trans_angle_deg is None else round(pr.trans_angle_deg, 4),
        "translation_scale_ratio": None if pr.trans_scale_ratio is None else round(pr.trans_scale_ratio, 4),
    }


def _candidate_brief(c: CandidateResult) -> dict[str, Any]:
    return {
        "order_name": c.order_name,
        "order_description": c.order_description,
        "overall_rms": None if math.isinf(c.overall_rms) else round(c.overall_rms, 6),
        "passed": c.passed,
        "error": c.error,
        "max_pair_rms": round(max((p.per_pair_rms for p in c.pairs), default=float("inf")), 6),
        "pairs": [
            {"pair_id": p.pair_id, "per_pair_rms": round(p.per_pair_rms, 6), "passed": p.passed}
            for p in c.pairs
        ],
    }


def _outcome_to_dict(outcome: CalibrationOutcome) -> dict[str, Any]:
    accepted = None
    if outcome.accepted is not None:
        c = outcome.accepted
        accepted = {
            "order_name": c.order_name,
            "order_description": c.order_description,
            "overall_rms": round(c.overall_rms, 6),
            "R": np.round(c.R, 8).tolist(),
            "T": np.round(c.T, 8).tolist(),
            "T_flat": np.round(c.T.ravel(), 8).tolist(),
            "pairs": [_pair_to_dict(p) for p in c.pairs],
            "selected_pair_ids": outcome.selected_pair_ids,
            "threshold": outcome.threshold,
        }
    return {
        "success": outcome.success,
        "reason": outcome.reason,
        "selected_pair_ids": outcome.selected_pair_ids,
        "threshold": outcome.threshold,
        "passing_candidates": outcome.passing_candidates,
        "accepted": accepted,
        "candidates": [_candidate_brief(c) for c in outcome.all_candidates],
    }


def _build_inputs(payload: CalibrationPayload) -> tuple[
    CameraParams, CameraParams, BoardSpec, list[dict], tuple[int, int]
]:
    Kl = np.asarray(payload.left_camera.camera_matrix, dtype=np.float64)
    Kr = np.asarray(payload.right_camera.camera_matrix, dtype=np.float64)
    if Kl.shape != (3, 3) or Kr.shape != (3, 3):
        raise HTTPException(422, "camera_matrix 必须是 3x3 矩阵")
    dl = np.asarray(payload.left_camera.dist_coeffs, dtype=np.float64).ravel()
    dr = np.asarray(payload.right_camera.dist_coeffs, dtype=np.float64).ravel()

    pairs: list[dict] = []
    sizes: list[tuple[int, int]] = []
    for p in payload.image_pairs:
        lp = np.asarray(p.left_pts, dtype=np.float64)
        rp = np.asarray(p.right_pts, dtype=np.float64)
        if lp.ndim != 2 or lp.shape[1] != 2 or rp.ndim != 2 or rp.shape[1] != 2:
            raise HTTPException(422, f"图对 {p.pair_id} 的角点必须是 [[x,y],...] 形式")
        if len(lp) != len(rp):
            raise HTTPException(422, f"图对 {p.pair_id} 左右角点数量不一致")
        pairs.append({
            "pair_id": p.pair_id,
            "left_pts": lp.tolist(),
            "right_pts": rp.tolist(),
        })
        if p.image_size is not None:
            w, h = p.image_size
            if w <= 0 or h <= 0:
                raise HTTPException(422, f"图对 {p.pair_id} 的 image_size 非法")
            sizes.append((int(w), int(h)))

    if sizes:
        image_size = sizes[0]
    else:
        all_pts = np.concatenate(
            [np.asarray(p["left_pts"]).reshape(-1, 2) for p in pairs]
            + [np.asarray(p["right_pts"]).reshape(-1, 2) for p in pairs],
            axis=0,
        )
        image_size = (
            int(math.ceil(all_pts[:, 0].max())) + 1,
            int(math.ceil(all_pts[:, 1].max())) + 1,
        )

    return (
        CameraParams(Kl, dl),
        CameraParams(Kr, dr),
        BoardSpec(payload.board.rows, payload.board.cols, payload.board.square_size),
        pairs,
        image_size,
    )


# ---- 路由 -------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """接收标定 JSON，解析并留存为本批数据，返回前端所需的回显内容。"""
    try:
        raw = json.loads((await file.read()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(422, f"无法解析 JSON 文件：{exc}") from exc

    try:
        payload = CalibrationPayload(**{
            **raw,
            # 上传阶段不强制预选，给默认值以通过模型校验，标定前可在页面调整
            "selected_pair_ids": raw.get("selected_pair_ids")
            or [p["pair_id"] for p in raw.get("image_pairs", [])],
            "pixel_threshold": raw.get("pixel_threshold", 1.0),
        })
    except Exception as exc:
        raise HTTPException(422, f"JSON 内容校验失败：{exc}") from exc

    STORE.reset_with(raw)
    session_id = STORE.session_id
    return JSONResponse({
        "session_id": session_id,
        "board": payload.board.model_dump(),
        "left_camera": payload.left_camera.model_dump(),
        "right_camera": payload.right_camera.model_dump(),
        "image_pairs": [
            {
                "pair_id": p.pair_id,
                "point_count": len(p.left_pts),
                "image_size": p.image_size,
                "left_bounds": _bounds(p.left_pts),
                "right_bounds": _bounds(p.right_pts),
            }
            for p in payload.image_pairs
        ],
        "selected_pair_ids": payload.selected_pair_ids,
        "pixel_threshold": payload.pixel_threshold,
        "last_success": None,
    })


def _bounds(pts: list[list[float]]) -> dict[str, float]:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return {
        "x_min": round(float(arr[:, 0].min()), 2),
        "x_max": round(float(arr[:, 0].max()), 2),
        "y_min": round(float(arr[:, 1].min()), 2),
        "y_max": round(float(arr[:, 1].max()), 2),
    }


@app.post("/api/calibrate")
def calibrate(req: CalibrateRequest) -> JSONResponse:
    payload = req.data
    left, right, board, pairs, image_size = _build_inputs(payload)
    try:
        outcome = run_calibration(
            left, right, board, pairs,
            payload.selected_pair_ids,
            payload.pixel_threshold,
            image_size,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    body = _outcome_to_dict(outcome)
    stale = False
    if outcome.success and outcome.accepted is not None:
        token = STORE.accept_calibration(
            body["accepted"],
            {
                "left_camera_matrix": np.asarray(
                    payload.left_camera.camera_matrix, np.float64
                ).tolist(),
                "right_camera_matrix": np.asarray(
                    payload.right_camera.camera_matrix, np.float64
                ).tolist(),
            },
            session_id=req.session_id,
        )
        # 换批后迟到的旧批次标定：不得写入新批次
        stale = req.session_id is not None and token is None
    if stale:
        body["success"] = False
        body["accepted"] = None
        body["reason"] = "批次已切换，本次标定结果已作废，未保存。"
        body["stale"] = True
    # 并发安全：返回本次请求看到的一致快照
    with STORE.lock:
        body["last_success"] = STORE.last_success
        body["calib_token"] = STORE.calib_token
    return JSONResponse(body)


@app.get("/api/last-success")
def last_success() -> JSONResponse:
    with STORE.lock:
        return JSONResponse({
            "last_success": STORE.last_success,
            "calib_token": STORE.calib_token,
            "range_result": STORE.range_result,
        })


# ---- 测点深度核验 -----------------------------------------------------------

_DEPTH_EPS_MM = 1e-6


def _verdict(range_dict: dict, min_depth: float, max_depth: float) -> dict[str, Any]:
    """只有整个深度区间都落在允许范围内才判可用，其余一律提示复测。"""
    status = range_dict["status"]
    if status == "no_solution":
        return {
            "usable": False,
            "code": "no_solution",
            "message": "无可行位置：在两台相机前方且同时落入左右图像素容差内的点不存在，请复测。",
        }
    z_min = range_dict["z_min_mm"]
    if status == "unbounded":
        return {
            "usable": False,
            "code": "unbounded",
            "message": (
                "深度无上界：像素容差叠加后的可行深度可以无限增大，"
                + ("且近端已小于允许下限，请复测。" if z_min < min_depth - _DEPTH_EPS_MM
                   else "无法确认远端是否超限，请复测。")
            ),
        }

    z_max = range_dict["z_max_mm"]
    below = z_min < min_depth - _DEPTH_EPS_MM
    above = z_max > max_depth + _DEPTH_EPS_MM
    if not below and not above:
        return {
            "usable": True,
            "code": "usable",
            "message": "整个深度区间都落在允许范围内。",
        }
    if z_max < min_depth - _DEPTH_EPS_MM:
        msg = "整个深度区间都小于允许下限（测点过近），请复测。"
    elif z_min > max_depth + _DEPTH_EPS_MM:
        msg = "整个深度区间都大于允许上限（测点过远），请复测。"
    elif below and above:
        msg = "深度区间同时越过允许下限和上限，请复测。"
    elif below:
        msg = "深度区间近端小于允许下限，请复测。"
    else:
        msg = "深度区间远端大于允许上限，请复测。"
    return {"usable": False, "code": "outside", "message": msg}


@app.post("/api/range")
def range_measure(req: RangeRequest) -> JSONResponse:
    """用本批最近一次成功标定的内外参，核验测点深度区间。"""
    with STORE.lock:
        session_id = STORE.session_id
        last_success = STORE.last_success
        calib_token = STORE.calib_token
        accepted_seq = STORE.range_seq

        if req.session_id != session_id:
            raise HTTPException(409, "批次已切换，请基于当前批次重新核验")
        if last_success is None or calib_token is None:
            raise HTTPException(409, "本批尚无成功标定结果，无法测距")
        if req.calib_token != calib_token:
            raise HTTPException(409, "标定结果已更新，请基于最新标定重新核验")
        if req.request_seq <= accepted_seq:
            # 迟到的旧请求：序号不高于已受理的序号，直接拒绝且不回填
            raise HTTPException(409, "请求序号过期，结果已作废")
        # 输入变化：先撤下旧测距结果，再计算
        STORE.range_seq = req.request_seq
        STORE.range_result = None

    if req.min_depth_mm >= req.max_depth_mm:
        raise HTTPException(422, "允许深度下限必须小于上限")
    if not all(math.isfinite(v) for v in (*req.left_px, *req.right_px)):
        raise HTTPException(422, "像素坐标必须是有限数")

    intrinsics = last_success["intrinsics"]
    accepted = last_success["result"]
    K_left = np.asarray(intrinsics["left_camera_matrix"], np.float64)
    K_right = np.asarray(intrinsics["right_camera_matrix"], np.float64)
    R = np.asarray(accepted["R"], np.float64)
    T = np.asarray(accepted["T_flat"], np.float64).reshape(3, 1)

    try:
        depth = compute_depth_range(
            K_left, K_right, R, T,
            req.left_px, req.right_px, req.tolerance_px,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    range_dict = {
        "status": depth.status,
        "z_min_mm": None if depth.z_min_mm is None else round(depth.z_min_mm, 4),
        "z_max_mm": None if depth.z_max_mm is None else round(depth.z_max_mm, 4),
        "tolerance_px": depth.tolerance_px,
    }
    verdict = _verdict(range_dict, req.min_depth_mm, req.max_depth_mm)
    body = {
        "session_id": session_id,
        "calib_token": calib_token,
        "request_seq": req.request_seq,
        "calibration_basis": {
            "order_name": accepted["order_name"],
            "selected_pair_ids": accepted["selected_pair_ids"],
            "overall_rms": accepted["overall_rms"],
            "threshold": accepted["threshold"],
        },
        "input": {
            "left_px": req.left_px,
            "right_px": req.right_px,
            "tolerance_px": req.tolerance_px,
            "min_depth_mm": req.min_depth_mm,
            "max_depth_mm": req.max_depth_mm,
        },
        "range": range_dict,
        "verdict": verdict,
        "computed_at": time.time(),
    }
    with STORE.lock:
        # 计算期间若有更新的请求、重新标定或换批，本结果视为迟到，不得写回
        if (STORE.range_seq != req.request_seq
                or STORE.calib_token != calib_token
                or STORE.session_id != session_id):
            raise HTTPException(409, "结果已过期，未保存")
        STORE.range_result = body
    return JSONResponse(body)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
