"""
Standalone LeRobot v3 数据集回放工具
不依赖 lerobot 包，只需要 cv2, pandas, pyarrow
支持显示 observation.tactile_left / observation.tactile_right 触觉数据

用法:
  python -m policy_adapters.replay_lerobot_v3 --dataset-root /path/to/lerobot_dataset
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TACTILE_LEFT_KEY = "observation.tactile_left"
TACTILE_RIGHT_KEY = "observation.tactile_right"
TACTILE_KEYS = (TACTILE_LEFT_KEY, TACTILE_RIGHT_KEY)

# PowerShell examples:
# paxini:
# python -m policy_adapters.replay_lerobot_v3 --dataset-root /path/to/paxini_lerobot_dataset

# tashan:
# python -m policy_adapters.replay_lerobot_v3 --dataset-root /path/to/tashan_lerobot_dataset

def load_dataset_info(dataset_root: Path) -> dict:
    """加载 meta/info.json"""
    with open(dataset_root / "meta" / "info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_stats(dataset_root: Path) -> dict:
    """加载 meta/stats.json；没有统计文件时返回空 dict。"""
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        return {}
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_parquet_data(dataset_root: Path, info: dict) -> pd.DataFrame:
    """加载 data/chunk-XXX/file-XXX.parquet"""
    # 按 info.json 中的 data_path 模式查找
    data_pattern = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    # 简化：扫描所有 parquet 文件
    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"在 {dataset_root / 'data'} 中找不到 .parquet 文件")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True)


def load_video_frames(dataset_root: Path, video_key: str, info: dict) -> list[np.ndarray]:
    """加载某个 camera 的所有视频帧"""
    video_dir = dataset_root / "videos" / video_key
    video_files = sorted(video_dir.rglob("*.mp4"))
    frames = []
    for vf in video_files:
        cap = cv2.VideoCapture(str(vf))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    return frames


def stack_vector_column(df: pd.DataFrame, col: str) -> np.ndarray:
    """把 parquet 中的 list/ndarray 向量列堆叠成 (N, D) float32 数组。"""
    raw_values = df[col].values
    if len(raw_values) == 0:
        return np.empty((0, 0), dtype=np.float32)
    first = np.asarray(raw_values[0], dtype=np.float32).reshape(-1)
    if first.size == 0:
        return np.empty((len(raw_values), 0), dtype=np.float32)
    return np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in raw_values])


def load_tactile_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """自动检测并加载左右 tactile 列。"""
    tactile_arrays = {}
    for key in TACTILE_KEYS:
        if key in df.columns:
            tactile_arrays[key] = stack_vector_column(df, key)
    return tactile_arrays


def normalize_tactile_values(values: np.ndarray, stats_entry: dict | None) -> tuple[np.ndarray, str]:
    """优先用 meta/stats.json 的逐维 min/max 归一化；缺失时用当前帧百分位。"""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if stats_entry and "min" in stats_entry and "max" in stats_entry:
        mins = np.asarray(stats_entry["min"], dtype=np.float32).reshape(-1)
        maxs = np.asarray(stats_entry["max"], dtype=np.float32).reshape(-1)
        if mins.shape == arr.shape and maxs.shape == arr.shape:
            denom = maxs - mins
            valid = np.abs(denom) > 1e-6
            norm = np.zeros_like(arr, dtype=np.float32)
            norm[valid] = (arr[valid] - mins[valid]) / denom[valid]
            return np.clip(norm, 0.0, 1.0), "stats"

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32), "empty"
    lo, hi = np.percentile(finite, [2, 98])
    if abs(float(hi) - float(lo)) < 1e-6:
        hi = lo + 1.0
    norm = (arr - float(lo)) / (float(hi) - float(lo))
    return np.clip(norm, 0.0, 1.0).astype(np.float32), "frame"


def infer_tactile_grid(values_len: int, profile: str, original_len: int) -> tuple[int, int]:
    """根据传感器类型推断热力图布局。"""
    profile = profile.lower()
    if original_len == 25 and values_len == 25:
        return 5, 5
    if profile == "paxini" and original_len == 234 and values_len == 231:
        return 11, 21

    root = int(math.sqrt(values_len))
    for rows in range(root, 0, -1):
        if values_len % rows == 0:
            return rows, values_len // rows
    return 1, values_len


def vector_to_grid(values: np.ndarray, profile: str, original_len: int) -> np.ndarray:
    """把一维 tactile 数据排成尽量规则的二维网格。"""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    rows, cols = infer_tactile_grid(values.size, profile, original_len)
    required = rows * cols
    if required == values.size:
        return values.reshape(rows, cols)

    cols = max(1, int(math.ceil(math.sqrt(values.size))))
    rows = int(math.ceil(values.size / cols))
    padded = np.zeros(rows * cols, dtype=np.float32)
    padded[: values.size] = values
    return padded.reshape(rows, cols)


def apply_tactile_colormap(norm_grid: np.ndarray) -> np.ndarray:
    """把 [0, 1] 热力图转成 BGR 伪彩色。"""
    gray = np.clip(norm_grid * 255.0, 0, 255).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray, colormap)


def draw_grid_lines(img: np.ndarray, rows: int, cols: int, x: int, y: int, w: int, h: int) -> None:
    """在小型热力图上画网格线，帮助看清 tactile taxel。"""
    if rows > 24 or cols > 32:
        return
    color = (35, 35, 35)
    for r in range(1, rows):
        yy = y + int(r * h / rows)
        cv2.line(img, (x, yy), (x + w, yy), color, 1)
    for c in range(1, cols):
        xx = x + int(c * w / cols)
        cv2.line(img, (xx, y), (xx, y + h), color, 1)


def draw_force_bars(panel: np.ndarray, force_values: np.ndarray, x: int, y: int, w: int) -> None:
    """Paxini 前 3 维 force 的简易条形显示。"""
    if force_values.size == 0:
        return
    max_abs = max(float(np.max(np.abs(force_values))), 1.0)
    cv2.putText(panel, "force[0:3]", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    for i, val in enumerate(force_values[:3]):
        yy = y + i * 18
        cv2.putText(panel, f"f{i}", (x, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        bar_x = x + 26
        bar_w = w - 92
        cv2.rectangle(panel, (bar_x, yy), (bar_x + bar_w, yy + 12), (65, 65, 65), -1)
        center_x = bar_x + bar_w // 2
        val_x = center_x + int(np.clip(float(val) / max_abs, -1.0, 1.0) * bar_w / 2)
        cv2.line(panel, (val_x, yy), (val_x, yy + 12), (0, 255, 255), 2)
        cv2.putText(panel, f"{float(val):.2f}", (bar_x + bar_w + 6, yy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (245, 245, 245), 1)


def render_tactile_panel(
    label: str,
    values: np.ndarray,
    stats_entry: dict | None,
    profile: str,
    panel_w: int = 560,
    panel_h: int = 240,
) -> np.ndarray:
    """渲染单侧 tactile 条状图面板。"""
    panel = np.full((panel_h, panel_w, 3), (22, 22, 22), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (75, 75, 75), 1)

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    norm, norm_mode = normalize_tactile_values(arr, stats_entry)

    skip = 3 if profile.lower() == "paxini" and arr.size == 234 else 0
    bar_values = norm[skip:]
    raw_bar_values = arr[skip:]

    cv2.putText(panel, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 235, 170), 2)
    cv2.putText(panel, f"len={arr.size} bars={bar_values.size} norm={norm_mode}",
                (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)

    chart_x, chart_y = 14, 66
    chart_w = panel_w - 28
    chart_h = panel_h - 104
    if skip:
        draw_force_bars(panel, arr[:skip], 14, chart_y + 12, panel_w - 28)
        chart_y += 72
        chart_h -= 58

    cv2.rectangle(panel, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (55, 55, 55), 1)
    if bar_values.size:
        baseline = chart_y + chart_h
        n = int(bar_values.size)
        for i, val in enumerate(bar_values):
            x0 = chart_x + int(i * chart_w / n)
            x1 = chart_x + int((i + 1) * chart_w / n)
            if x1 <= x0:
                x1 = x0 + 1
            bar_h = int(float(np.clip(val, 0.0, 1.0)) * (chart_h - 2))
            color = (
                60,
                int(115 + 120 * float(np.clip(val, 0.0, 1.0))),
                int(255 - 110 * float(np.clip(val, 0.0, 1.0))),
            )
            cv2.rectangle(panel, (x0, baseline - bar_h), (x1 - 1, baseline - 1), color, -1)

        tick_step = 5 if n <= 40 else 25
        for i in range(0, n, tick_step):
            x = chart_x + int(i * chart_w / max(n, 1))
            cv2.line(panel, (x, baseline), (x, baseline + 4), (130, 130, 130), 1)
            cv2.putText(panel, str(i + skip), (x + 2, min(panel_h - 38, baseline + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 160, 160), 1)

    raw_min = float(np.min(arr)) if arr.size else 0.0
    raw_mean = float(np.mean(arr)) if arr.size else 0.0
    raw_max = float(np.max(arr)) if arr.size else 0.0
    bar_min = float(np.min(raw_bar_values)) if raw_bar_values.size else 0.0
    bar_max = float(np.max(raw_bar_values)) if raw_bar_values.size else 0.0
    stats_text = f"raw min/mean/max: {raw_min:.3g} / {raw_mean:.3g} / {raw_max:.3g}  bar range: {bar_min:.3g}..{bar_max:.3g}"
    cv2.putText(panel, stats_text, (14, panel_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)

    return panel


def hstack_padded(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    """水平拼接不同高度图像。"""
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    max_h = max(img.shape[0] for img in images)
    padded = []
    for i, img in enumerate(images):
        if img.shape[0] < max_h:
            pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)
        if gap and i < len(images) - 1:
            padded.append(np.zeros((max_h, gap, 3), dtype=np.uint8))
    return np.hstack(padded)


def vstack_padded(images: list[np.ndarray], gap: int = 0) -> np.ndarray:
    """垂直拼接不同宽度图像。"""
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    max_w = max(img.shape[1] for img in images)
    padded = []
    for i, img in enumerate(images):
        if img.shape[1] < max_w:
            pad = np.zeros((img.shape[0], max_w - img.shape[1], 3), dtype=np.uint8)
            img = np.hstack([img, pad])
        padded.append(img)
        if gap and i < len(images) - 1:
            padded.append(np.zeros((gap, max_w, 3), dtype=np.uint8))
    return np.vstack(padded)


def render_tactile_row(
    tactile_arrays: dict[str, np.ndarray],
    frame_idx: int,
    stats: dict,
    profile: str,
) -> np.ndarray | None:
    """渲染左右 tactile 面板；没有 tactile 时返回 None。"""
    panels = []
    labels = {
        TACTILE_LEFT_KEY: "tactile_left",
        TACTILE_RIGHT_KEY: "tactile_right",
    }
    for key in TACTILE_KEYS:
        if key not in tactile_arrays:
            continue
        values = tactile_arrays[key][frame_idx]
        panel = render_tactile_panel(labels[key], values, stats.get(key), profile)
        panels.append(panel)
    if not panels:
        return None
    return hstack_padded(panels, gap=8)


def append_status_bar(canvas: np.ndarray, text: str, height: int = 28) -> np.ndarray:
    """在画布底部追加状态栏，避免覆盖视频或 tactile 面板。"""
    bar = np.zeros((height, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1)
    return np.vstack([canvas, bar])


def draw_vector_bars(
    panel: np.ndarray,
    label: str,
    values: np.ndarray,
    x: int,
    y: int,
    width: int,
    value_scale: float,
    color: tuple[int, int, int],
    value_fmt: str = "{:.4f}",
) -> None:
    """在信息面板里绘制一组一维数值条。"""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    cv2.putText(panel, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    bar_x = x + 52
    bar_w = max(80, width - 150)
    center_x = bar_x + bar_w // 2
    row_h = 23
    for i, val in enumerate(arr):
        yy = y + i * row_h
        cv2.putText(panel, f"{i}", (x, yy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (185, 185, 185), 1)
        cv2.rectangle(panel, (bar_x, yy), (bar_x + bar_w, yy + 14), (60, 60, 60), -1)
        cv2.line(panel, (center_x, yy), (center_x, yy + 14), (105, 105, 105), 1)
        display_val = float(np.clip(float(val) / value_scale, -1.0, 1.0))
        val_x = center_x + int(display_val * bar_w / 2)
        cv2.line(panel, (val_x, yy), (val_x, yy + 14), color, 2)
        cv2.putText(panel, value_fmt.format(float(val)), (bar_x + bar_w + 8, yy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (235, 235, 235), 1)


def render_state_action_panel(
    state_values: np.ndarray | None,
    actions: np.ndarray | None,
    frame_idx: int,
    width: int,
) -> np.ndarray:
    """单独渲染 state/action，避免覆盖相机图像。"""
    state_dim = int(np.asarray(state_values).reshape(-1).size) if state_values is not None else 0
    action_values = actions[frame_idx] if actions is not None and actions.ndim == 2 else actions
    action_dim = int(np.asarray(action_values).reshape(-1).size) if action_values is not None else 0
    rows = max(state_dim, action_dim, 1)
    panel_h = 42 + rows * 23
    panel = np.full((panel_h, width, 3), (18, 18, 18), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, panel_h - 1), (70, 70, 70), 1)
    cv2.putText(panel, "state / action", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1)

    col_w = max(360, (width - 42) // 2)
    if state_values is not None:
        draw_vector_bars(
            panel,
            "state",
            state_values,
            14,
            58,
            col_w,
            value_scale=3.0,
            color=(90, 210, 255),
        )
    if action_values is not None:
        draw_vector_bars(
            panel,
            "action",
            action_values,
            28 + col_w,
            58,
            col_w,
            value_scale=1.0,
            color=(0, 235, 170),
        )
    return panel


def render_camera_panel(label: str, frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """渲染单个相机，不在图像上覆盖标签。"""
    img = resize_to_fit(frame, max_w, max_h)
    header_h = 30
    panel = np.zeros((img.shape[0] + header_h, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 235, 170), 1)
    panel[header_h:, :] = img
    return panel


def ordered_video_keys(video_keys: list[str]) -> list[str]:
    """让 front/wrist 优先横向并列显示。"""
    preferred = ["observation.images.front", "observation.images.wrist", "front", "wrist"]
    ordered = [key for name in preferred for key in video_keys if key == name]
    ordered.extend([key for key in video_keys if key not in ordered])
    return ordered


def render_video_row(video_frames: dict[str, list[np.ndarray]], video_keys: list[str], frame_idx: int) -> np.ndarray:
    """将 front/wrist 等相机独立并列放置。"""
    panels = []
    for key in ordered_video_keys(video_keys):
        frame = video_frames[key][frame_idx].copy()
        panels.append(render_camera_panel(key, frame, max_w=620, max_h=390))
    return hstack_padded(panels, gap=8)


def draw_action_bar(img: np.ndarray, actions: np.ndarray, frame_idx: int, 
                    y_offset: int = 30, bar_width: int = 400, bar_height: int = 20) -> np.ndarray:
    """在图像底部绘制动作值条形图"""
    h, w = img.shape[:2]
    overlay = img.copy()
    # 半透明背景
    n_dims = actions.shape[1] if actions.ndim == 2 else len(actions)
    panel_h = 40 + n_dims * 30
    cv2.rectangle(overlay, (10, h - panel_h - 10), (w - 10, h - 10), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    action = actions[frame_idx] if actions.ndim == 2 else actions
    for i, val in enumerate(action):
        y = h - panel_h + 10 + i * 30
        # 标签
        cv2.putText(img, f"a[{i}]", (20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        # 条形图背景
        bar_x = 80
        cv2.rectangle(img, (bar_x, y), (bar_x + bar_width, y + bar_height), (80, 80, 80), -1)
        # 值指示（归一化到 [-1, 1] 或直接显示）
        display_val = np.clip(float(val), -1.0, 1.0)
        center_x = bar_x + bar_width // 2
        val_x = center_x + int(display_val * bar_width / 2)
        cv2.line(img, (val_x, y), (val_x, y + bar_height), (0, 255, 255), 3)
        # 数值文本
        cv2.putText(img, f"{float(val):.4f}", (bar_x + bar_width + 10, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def resize_to_fit(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """缩放图像以适应屏幕"""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h))
    return img


def main():
    parser = argparse.ArgumentParser(description="LeRobot v3 数据集回放")
    parser.add_argument("--dataset-root", type=Path, required=True, help="数据集根目录")
    parser.add_argument("--fps", type=float, default=None, help="回放帧率 (默认从 info.json 读取)")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--no-tactile", action="store_true", help="不显示 tactile_left/right 触觉面板")
    args = parser.parse_args()

    root = args.dataset_root
    info = load_dataset_info(root)
    stats = load_dataset_stats(root)
    tactile_profile = str(info.get("tactile_profile", "")).lower()
    print(f"数据集: {info.get('task', 'unknown')}")
    print(f"Episodes: {info.get('total_episodes')}, Frames: {info.get('total_frames')}, FPS: {info.get('fps')}")
    if tactile_profile:
        print(f"Tactile profile: {tactile_profile}")

    # 加载 parquet 数据
    df = load_parquet_data(root, info)
    print(f"Parquet 列: {list(df.columns)}")
    print(f"Parquet 行数: {len(df)}")

    # 自动检测 action 列，并转换为 numpy array
    action_cols = [c for c in df.columns if "action" in c.lower()]
    print(f"Action 列: {action_cols}")

    # 解析 action —— 可能是嵌套 list，需要 stack
    if action_cols:
        action_col = action_cols[0]
        raw_actions = df[action_col].values
        # 检查是否是嵌套 list
        if isinstance(raw_actions[0], (list, np.ndarray)):
            actions = np.stack(raw_actions)  # (N, action_dim)
        else:
            actions = raw_actions.reshape(-1, 1)
        print(f"Action shape: {actions.shape}")
    else:
        actions = None

    state_values = stack_vector_column(df, "observation.state") if "observation.state" in df.columns else None
    if state_values is not None:
        print(f"State shape: {state_values.shape}")

    # 自动检测 tactile 列，并转换为 numpy array
    tactile_arrays = {} if args.no_tactile else load_tactile_arrays(df)
    if tactile_arrays:
        print("Tactile 列:")
        for key, arr in tactile_arrays.items():
            print(f"  {key}: shape={arr.shape}")
    elif args.no_tactile:
        print("Tactile 显示: disabled")
    else:
        print("Tactile 列: 未找到 observation.tactile_left/right")

    # 加载视频
    video_keys = [d.name for d in (root / "videos").iterdir() if d.is_dir()]
    print(f"视频通道: {video_keys}")

    video_frames = {}
    for vk in video_keys:
        frames = load_video_frames(root, vk, info)
        video_frames[vk] = frames
        print(f"  {vk}: {len(frames)} 帧")

    total_frames = min(len(df), *[len(v) for v in video_frames.values()])
    print(f"可用总帧数: {total_frames}")

    fps = args.fps or info.get("fps", 10)
    delay_ms = int(1000 / fps)

    frame_idx = args.start_frame
    paused = False

    cv2.namedWindow("Dataset Replay", cv2.WINDOW_NORMAL)

    while 0 <= frame_idx < total_frames:
        canvas = render_video_row(video_frames, video_keys, frame_idx)

        state_row = render_state_action_panel(
            state_values[frame_idx] if state_values is not None else None,
            actions,
            frame_idx,
            width=canvas.shape[1],
        )
        canvas = vstack_padded([canvas, state_row], gap=8)

        tactile_row = render_tactile_row(tactile_arrays, frame_idx, stats, tactile_profile)
        if tactile_row is not None:
            canvas = vstack_padded([canvas, tactile_row], gap=8)

        # 帧数信息
        canvas = append_status_bar(
            canvas,
            f"Frame: {frame_idx}/{total_frames}  {'[PAUSED]' if paused else ''}"
        )

        cv2.imshow("Dataset Replay", canvas)

        key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('q'):
            break
        elif key == ord(' '):  # 空格暂停/播放
            paused = not paused
        elif key == ord('a') or key == 81:  # 左箭头 / A
            frame_idx = max(0, frame_idx - 1)
        elif key == ord('d') or key == 83:  # 右箭头 / D
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif key == ord('w'):  # 快退 30 帧
            frame_idx = max(0, frame_idx - 30)
        elif key == ord('s'):  # 快进 30 帧
            frame_idx = min(total_frames - 1, frame_idx + 30)
        elif key == ord('r'):  # 重置到开头
            frame_idx = 0
        elif key == ord('f'):  # 跳到末尾
            frame_idx = total_frames - 1
        else:
            # 自动播放
            if not paused:
                frame_idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
