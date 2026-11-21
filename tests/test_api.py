"""FastAPI 端到端测试：上传、标定成功/失败、上次成功结果留存与批次切换。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import STORE, app

client = TestClient(app)
SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_calibration.json"


def _upload() -> dict:
    with SAMPLE.open("rb") as fh:
        resp = client.post(
            "/api/upload",
            files={"file": ("sample_calibration.json", fh, "application/json")},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _calibrate(raw: dict, ids: list[str], threshold: float):
    body = {
        "data": {
            **raw,
            "selected_pair_ids": ids,
            "pixel_threshold": threshold,
        }
    }
    return client.post("/api/calibrate", json=body)


def test_health_and_index():
    assert client.get("/api/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200 and "双目" in page.text
    js = client.get("/static/app.js")
    assert js.status_code == 200


def test_upload_rejects_bad_json():
    resp = client.post(
        "/api/upload",
        files={"file": ("bad.json", b"{not json", "application/json")},
    )
    assert resp.status_code == 422


def test_full_flow_success_then_failure_keeps_last_result():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    assert len(meta["image_pairs"]) == 6
    assert STORE.last_success is None  # 新批次必须清空旧结果

    good_ids = [p["pair_id"] for p in raw["image_pairs"][:5]]
    ok = _calibrate(raw, good_ids, 0.6)
    assert ok.status_code == 200
    ok_body = ok.json()
    assert ok_body["success"] is True
    assert ok_body["accepted"]["order_name"] == "flip_cols"
    assert ok_body["accepted"]["selected_pair_ids"] == good_ids
    assert len(ok_body["accepted"]["pairs"]) == 5

    # 上次成功结果已留存，且记录使用了哪些图对
    kept = client.get("/api/last-success").json()["last_success"]
    assert kept["result"]["selected_pair_ids"] == good_ids

    # 失败标定：含大噪声 pair_06，严格阈值
    all_ids = [p["pair_id"] for p in raw["image_pairs"]]
    fail = _calibrate(raw, all_ids, 0.6)
    assert fail.status_code == 200
    fail_body = fail.json()
    assert fail_body["success"] is False
    assert fail_body["accepted"] is None
    assert "没有任何一种角点排序" in fail_body["reason"]
    # 关键语义：失败后上次成功结果仍在，且标明所用图对
    assert fail_body["last_success"]["result"]["selected_pair_ids"] == good_ids
    assert client.get("/api/last-success").json()["last_success"] is not None


def test_ambiguous_calibration_does_not_overwrite():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    good_ids = [p["pair_id"] for p in raw["image_pairs"][:4]]
    _calibrate(raw, good_ids, 0.6)  # 先建立成功结果

    loose = _calibrate(raw, good_ids, 100.0)
    body = loose.json()
    assert body["success"] is False
    assert len(body["passing_candidates"]) >= 2
    # 多解时同样保留上次成功结果
    assert body["last_success"] is not None
    assert body["last_success"]["result"]["order_name"] == "flip_cols"


def test_reupload_resets_last_success():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    good_ids = [p["pair_id"] for p in raw["image_pairs"][:4]]
    assert _calibrate(raw, good_ids, 0.6).json()["success"] is True
    assert client.get("/api/last-success").json()["last_success"] is not None

    # 重新上传一批 -> 上一批结果清空
    _upload()
    assert client.get("/api/last-success").json()["last_success"] is None


def test_less_than_three_pairs_rejected():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    resp = _calibrate(raw, ["pair_01", "pair_02"], 0.6)
    assert resp.status_code == 422


def test_invalid_threshold_rejected():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    resp = _calibrate(
        raw, [p["pair_id"] for p in raw["image_pairs"][:4]], 0
    )
    assert resp.status_code == 422


def test_upload_with_only_two_pairs_rejected():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    raw["image_pairs"] = raw["image_pairs"][:2]
    resp = client.post(
        "/api/upload",
        files={"file": ("short.json", json.dumps(raw).encode(), "application/json")},
    )
    assert resp.status_code == 422


def test_non_square_camera_matrix_rejected():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    raw["left_camera"]["camera_matrix"] = [[1, 0, 0], [0, 1, 0]]  # 2x3
    resp = client.post(
        "/api/upload",
        files={"file": ("badk.json", json.dumps(raw).encode(), "application/json")},
    )
    # 上传阶段模型校验通过形状字段，3x3 形状在标定入口拦截
    assert resp.status_code == 200
    cal = _calibrate(
        raw, [p["pair_id"] for p in raw["image_pairs"][:4]], 0.6
    )
    assert cal.status_code == 422
    assert "3x3" in cal.json()["detail"]


# ---- 测点深度核验 -----------------------------------------------------------

def _probe_pixels(raw: dict) -> tuple[list[float], list[float]]:
    return raw["depth_probe"]["left_px"], raw["depth_probe"]["right_px"]


def _range_req(raw: dict, meta: dict, calib: dict, seq: int = 1, **overrides):
    lp, rp = _probe_pixels(raw)
    probe = raw["depth_probe"]
    body = {
        "session_id": meta["session_id"],
        "calib_token": calib.get("calib_token"),
        "request_seq": seq,
        "left_px": lp,
        "right_px": rp,
        "tolerance_px": probe["tolerance_px"],
        "min_depth_mm": probe["min_depth_mm"],
        "max_depth_mm": probe["max_depth_mm"],
    }
    body.update(overrides)
    return body


def _good_calibration(raw: dict):
    ids = [p["pair_id"] for p in raw["image_pairs"][:4]]
    resp = _calibrate(raw, ids, 0.6)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True, body
    return body


def test_range_usable_with_last_successful_calibration():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    calib = _good_calibration(raw)

    resp = client.post("/api/range", json=_range_req(raw, meta, calib, seq=1))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["range"]["status"] == "bounded"
    truth_z = raw["_ground_truth"]["probe_point_left_mm"][2]
    assert body["range"]["z_min_mm"] < truth_z < body["range"]["z_max_mm"]
    assert body["verdict"]["usable"] is True
    assert body["calibration_basis"]["order_name"] == "flip_cols"
    # 结果被留存
    kept = client.get("/api/last-success").json()
    assert kept["range_result"]["request_seq"] == 1


def test_range_requires_calibration():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    resp = client.post(
        "/api/range",
        json=_range_req(raw, meta, {"calib_token": "nope"}, seq=1),
    )
    assert resp.status_code == 409
    assert "尚无成功标定" in resp.json()["detail"]


def test_failed_calibration_keeps_ranging_usable():
    """试标失败后，仍可用本批保留的标定继续测距。"""
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    calib = _good_calibration(raw)

    all_ids = [p["pair_id"] for p in raw["image_pairs"]]
    fail = _calibrate(raw, all_ids, 0.6).json()
    assert fail["success"] is False
    assert fail["last_success"]["calib_token"] == calib["calib_token"]

    # 用失败响应里返回的留存基准仍能测距
    resp = client.post(
        "/api/range",
        json=_range_req(raw, meta, {"calib_token": fail["last_success"]["calib_token"]}, seq=1),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["verdict"]["usable"] is True


def test_recalibration_invalidates_old_token():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    first = _good_calibration(raw)
    _good_calibration(raw)  # 再标一次，签发新 token
    resp = client.post(
        "/api/range",
        json=_range_req(raw, meta, {"calib_token": first["calib_token"]}, seq=1),
    )
    assert resp.status_code == 409
    assert "标定结果已更新" in resp.json()["detail"]


def test_reupload_invalidates_session():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    calib = _good_calibration(raw)
    old_session = meta["session_id"]
    _upload()  # 换批
    resp = client.post(
        "/api/range",
        json=_range_req(raw, {"session_id": old_session}, calib, seq=1),
    )
    assert resp.status_code == 409
    assert "批次已切换" in resp.json()["detail"]


def test_stale_request_seq_rejected_and_not_backfilled():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    calib = _good_calibration(raw)
    ok = client.post("/api/range", json=_range_req(raw, meta, calib, seq=2))
    assert ok.status_code == 200
    # 迟到的旧序号：拒绝，且缓存仍是 seq=2 的结果
    stale = client.post("/api/range", json=_range_req(raw, meta, calib, seq=1))
    assert stale.status_code == 409
    assert client.get("/api/last-success").json()["range_result"]["request_seq"] == 2


def test_range_new_input_clears_previous_result():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    calib = _good_calibration(raw)
    lp, rp = _probe_pixels(raw)
    first = client.post("/api/range", json=_range_req(raw, meta, calib, seq=1))
    assert first.json()["verdict"]["usable"] is True
    # 新输入：右图像素错位 -> 无解，旧结果被撤下
    bad = client.post(
        "/api/range",
        json=_range_req(raw, meta, calib, seq=2,
                        right_px=[rp[0] - 400.0, rp[1]]),
    )
    assert bad.status_code == 200
    assert bad.json()["range"]["status"] == "no_solution"
    assert bad.json()["verdict"]["usable"] is False
    kept = client.get("/api/last-success").json()["range_result"]
    assert kept["request_seq"] == 2
    assert kept["range"]["status"] == "no_solution"


def test_range_unbounded_is_not_usable():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    calib = _good_calibration(raw)
    lp, rp = _probe_pixels(raw)
    body = _range_req(raw, meta, calib, seq=1, right_px=[rp[0], rp[1]], tolerance_px=80.0)
    resp = client.post("/api/range", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["range"]["status"] == "unbounded"
    assert data["range"]["z_max_mm"] is None
    assert data["verdict"]["usable"] is False


@pytest.mark.parametrize("overrides,detail", [
    ({"min_depth_mm": 1000, "max_depth_mm": 1000}, "下限必须小于上限"),
    ({"min_depth_mm": -1}, "422"),
    ({"tolerance_px": -0.5}, "422"),
])
def test_range_input_validation(overrides, detail):
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    calib = _good_calibration(raw)
    resp = client.post("/api/range", json=_range_req(raw, meta, calib, seq=1, **overrides))
    assert resp.status_code == 422
    if detail != "422":
        assert detail in resp.json()["detail"]


def test_range_thread_safe_against_late_result():
    """模拟计算期间出现更新请求：迟到的结果不得写入缓存。"""
    from app.main import STORE

    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    calib = _good_calibration(raw)
    lp, rp = _probe_pixels(raw)

    # 直接构造“计算前序号已被更高请求占位”的情形
    STORE.range_seq = 5
    resp = client.post(
        "/api/range",
        json=_range_req(raw, meta, calib, seq=4),
    )
    assert resp.status_code == 409
    assert STORE.range_result is None


def test_concurrent_requests_only_latest_written():
    """并发发出两个测距请求：序号较小者不得覆盖序号较大者的结果。"""
    import threading

    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    calib = _good_calibration(raw)
    lp, rp = _probe_pixels(raw)

    barrier = threading.Barrier(2)
    results = {}

    def worker(seq, tol):
        barrier.wait()
        resp = client.post(
            "/api/range",
            json=_range_req(raw, meta, calib, seq=seq, tolerance_px=tol),
        )
        results[seq] = resp.status_code

    threads = [threading.Thread(target=worker, args=(11, 1.0)),
               threading.Thread(target=worker, args=(12, 3.0))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert set(results.values()) <= {200, 409}
    assert results[12] == 200  # 较大序号必然先通过序号校验
    from app.main import STORE
    if STORE.range_result is not None:
        assert STORE.range_result["request_seq"] == 12


def test_late_calibration_from_previous_batch_rejected():
    """换批后，上一批在途的标定响应不得写入新批次。"""
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    old_meta = _upload()
    new_meta = _upload()  # 期间换批
    ids = [p["pair_id"] for p in raw["image_pairs"][:4]]
    late = client.post("/api/calibrate", json={
        "data": {**raw, "selected_pair_ids": ids, "pixel_threshold": 0.6},
        "session_id": old_meta["session_id"],
    })
    assert late.status_code == 200
    body = late.json()
    assert body["success"] is False
    assert body["stale"] is True
    assert body["last_success"] is None
    assert client.get("/api/last-success").json()["last_success"] is None
    assert new_meta["session_id"] != old_meta["session_id"]
