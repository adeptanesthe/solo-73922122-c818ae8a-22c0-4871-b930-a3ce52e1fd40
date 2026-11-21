# 双目相机外参标定演示（FastAPI + OpenCV）

双目检测工位更换右相机后，重新标定 **左相机 → 右相机** 的相对位姿（旋转 R、平移 T）。
维修员在网页上传一份 JSON（两台相机内参、棋盘格行列数与格距、3～6 对同步图的
去畸变角点），勾选至少 3 对图并设定像素误差上限即可试标。

标定成功后，可用同一页面第 5 节 **核验工件上某个测点是否处在双目工位的工作距离内**：
填入该点在左右图的去畸变像素坐标、统一像素容差和允许深度上下限，得到测点
**沿左相机光轴的深度区间（毫米）** 与可用判定。

## 核心判定规则

1. **左图角点顺序为基准**。右图角点列表可能与左图整体顺序不一致（整体倒序、
   旋转等），后端枚举棋盘格的全部对称排序（非方形 4 种、方形 8 种）逐一验证。
2. 对每种候选排序，用所勾选的全部图对做一次固定内参的立体标定
   （`cv2.stereoCalibrate` + `CALIB_FIX_INTRINSIC`），得到 **同一组外参 R,T**
   及每对图在该外参下的逐对重投影 RMSE。
3. 仅当 **恰好一种** 排序满足“所有被选图对 RMSE 均 ≤ 像素误差上限”时，
   才接受并更新结果：
   - **零种达标** → 报错说明原因，不更新，保留本批上次成功结果；
   - **多种达标** → 数据不足以唯一确定顺序，不更新，保留上次成功结果；
   - 页面始终标明上次成功结果 **使用了哪些图对、哪种排序、误差多少**。
4. 重新上传一批新数据时，上一批的成功结果自动清空。

## 测点深度核验规则

用 **本批最近一次成功标定** 的内外参，判断测点是否落在允许深度范围内（单位 mm，
深度沿左相机光轴 Z 计量）：

- 输入：同一点在左、右图的 **去畸变** 像素坐标 `(u,v)`，一个 **统一的非负像素容差**
  `±tol`，以及允许的深度上下限 `[Dmin, Dmax]`。
- 只考虑同时满足以下条件的位置：位于 **两台相机前方**，且投影到左右两图后
  横、纵坐标都落在 `±tol` 内。可行位置集合在 Z 轴上的投影即深度区间。
- 区间求解是精确的（投影不等式在相机前方可化为关于三维点的线性不等式，
  可行域为多面体），结果分三种状态：
  - `bounded`：`[z_min, z_max]` 有限；
  - `unbounded`：深度 **无上界**（`z_max = null`），例如左右像素对大容差时
    可容忍到无穷远；
  - `no_solution`：**无可行位置**（左右像素在对极几何上不匹配、或所需位置在
    相机后方）。
- 判定：**只有整个深度区间都落在 `[Dmin, Dmax]` 内才判“可用”**；
  `no_solution`、`unbounded`、区间部分或全部越界一律提示 **复测**。
- 状态失效规则：
  - 测距输入（像素、容差、深度上下限）一变化 → 撤下旧测距结果；
  - 换批（重新上传）或重新标定成功 → 撤下旧测距结果，旧 `calib_token` 失效；
  - 每次测距请求带单调递增 `request_seq`，序号过期或标定已变的响应一律丢弃，
    **迟到的旧结果不得回填**；
  - 标定请求同样携带 `session_id`：换批后才返回的旧批次标定结果会被后端拒收；
  - **试标失败后仍可用本批保留的标定测距**（旧结果只有在重标成功时才作废）。

## 目录结构

```
.
├── app/
│   ├── calibration.py        # 核心算法：排序枚举、立体标定、逐对误差、唯一性判定
│   ├── ranging.py            # 测点深度区间：半空间可行域 + 顶点/回收锥
│   ├── main.py               # FastAPI 路由、JSON 校验、上次结果与测距结果留存
│   └── static/
│       ├── index.html        # 单页前端（原生 JS）
│       ├── app.js
│       ├── style.css
│       └── sample_calibration.json   # 可一键载入的试标样例
├── samples/
│   └── sample_calibration.json       # 同一份样例（curl 上传用）
├── scripts/make_sample.py    # 样例生成脚本（真实投影 + 噪声，含真值 _ground_truth）
├── tests/                    # pytest：核心算法、测距几何、API 端到端
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 用 Docker Compose 启动（推荐）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

页面上点 **“载入内置示例”** 即可直接试标，无需自己准备数据。

命令行上传样例：

```bash
curl -F "file=@samples/sample_calibration.json;type=application/json" \
  http://localhost:8000/api/upload
```

## 本地直接运行（无需 Docker）

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 输入 JSON 格式

```json
{
  "left_camera":  { "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]], "dist_coeffs": [0,0,0,0,0] },
  "right_camera": { "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]], "dist_coeffs": [0,0,0,0,0] },
  "board": { "rows": 7, "cols": 5, "square_size": 25.0 },
  "image_pairs": [
    {
      "pair_id": "pair_01",
      "image_size": [1280, 960],
      "left_pts":  [[x, y], "...35 个角点"],
      "right_pts": [[x, y], "...顺序可能整体倒序"]
    }
  ],
  "selected_pair_ids": ["pair_01", "pair_02", "pair_03", "pair_04"],
  "pixel_threshold": 0.6
}
```

- 角点为 **已去畸变** 的像素坐标；`dist_coeffs` 可传全 0。
- `image_pairs` 数量 3～6；`selected_pair_ids` 至少 3 个；`pixel_threshold > 0`。
- `image_size` 可省略，缺省时由角点坐标范围推断。

## 内置样例说明

`scripts/make_sample.py` 用已知内参与已知左→右 R/T 正投影生成 6 对图：

| 图对 | 用途 |
|---|---|
| pair_01～pair_05 | 正常数据（0.12 px 检测噪声），右图角点为 **整体列倒序 flip_cols** |
| pair_06 | 注入 1.8 px 大噪声，用于演示“无排序达标 → 保留上次结果” |

推荐试标路径：

1. 选前 4 或前 5 对、阈值 **0.6 px** → 成功，唯一排序 `flip_cols`，
   R/T 与 JSON 内 `_ground_truth` 字段高度一致；
2. 勾选全部 6 对（含 pair_06）、阈值 0.6 → 无排序达标，页面报错且
   保留第 1 步的成功结果并标明所用图对；
3. 选前 4 对、阈值改成 **100 px** → 3 种排序同时达标（歧义），拒绝更新。

### 测点测距的试算数据

样例 JSON 内含 `depth_probe` 字段，点 **“填入样例测点”** 即可载入：

```json
"depth_probe": {
  "left_px":  [674.3478, 462.3261],
  "right_px": [426.0071, 451.9206],
  "tolerance_px": 1.0,
  "min_depth_mm": 850.0,
  "max_depth_mm": 1000.0
}
```

该测点的真实位置是左相机坐标系下的 `(30, -20, 920) mm`（见
`_ground_truth.probe_point_left_mm`），像素为真值投影、无噪声。

- 按第 1 步标定成功后核验 → 深度区间约 **[904.3, 936.6] mm**，**可用**；
- 容差改成 **0.2 px** → 区间收缩到约 **[917.0, 923.4] mm**，仍包住真值 920 mm；
- 容差改成 **0** → 标定外参的微小误差使左右像素无法严格一致，判
  **无可行位置**（真实标定结果下零容差通常无解，属正常现象）；
- 容差改成 **80 px** → **无上界**，判复测；
- 把右图像素 u 改小 400 px → **无可行位置**，判复测；
- 深度上下限改成 `[960, 1000]` → 区间整体小于下限，判复测；
- 深度上下限改成 `[900, 1000]`、容差 **40 px** → 区间约 [539.6, 3048.4]，
  两端同时越界，判复测。

重新生成样例：

```bash
python scripts/make_sample.py samples/sample_calibration.json
cp samples/sample_calibration.json app/static/sample_calibration.json
```

## API 摘要

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/upload` | multipart 上传标定 JSON，建立新批次并清空旧结果与旧测距 |
| POST | `/api/calibrate` | 执行标定，返回 R/T、每对排序与误差、全部候选排序诊断 |
| POST | `/api/range` | 用本批最近成功标定求测点深度区间与可用判定 |
| GET  | `/api/last-success` | 查看本批上次成功结果（含所用图对）与最新测距结果 |
| GET  | `/api/health` | 健康检查 |

`POST /api/range` 请求体：

```json
{
  "session_id": "上传返回的批次号",
  "calib_token": "标定成功响应里的 calib_token",
  "request_seq": 1,
  "left_px":  [674.3478, 462.3261],
  "right_px": [426.0071, 451.9206],
  "tolerance_px": 1.0,
  "min_depth_mm": 850.0,
  "max_depth_mm": 1000.0
}
```

批次已切换、标定已更新、`request_seq` 过期时返回 **409**；参数非法返回 **422**。

## 运行测试

```bash
pytest tests/ -v
```

## 备注

- 结果保存在进程内存中（演示用途）；多副本部署或重启后不持久化。
- OpenCV 4.x 与 5.x 的 `stereoCalibrate` 返回签名不同，代码已兼容
  （5.x 无 perViewErrors，逐对误差在应用层重投影计算）。
