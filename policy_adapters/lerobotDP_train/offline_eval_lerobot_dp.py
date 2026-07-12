from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from ..utils.common import (
        ACTION_KEY,
        FRONT_IMAGE_KEY,
        STATE_KEY,
        TACTILE_LEFT_KEY,
        TACTILE_RIGHT_KEY,
        WRIST_IMAGE_KEY,
    )
    from ..utils.action_space import ACTION_NAMES
    from .deploy_lerobot_dp_runtime import LeRobotDPRuntime
    from .train_lerobot_dp_minimal import _add_lerobot_to_path, _install_lerobot_namespace_shims
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from policy_adapters.utils.common import (  # type: ignore
        ACTION_KEY,
        FRONT_IMAGE_KEY,
        STATE_KEY,
        TACTILE_LEFT_KEY,
        TACTILE_RIGHT_KEY,
        WRIST_IMAGE_KEY,
    )
    from policy_adapters.utils.action_space import ACTION_NAMES  # type: ignore
    from policy_adapters.lerobotDP_train.deploy_lerobot_dp_runtime import (  # type: ignore
        LeRobotDPRuntime,
    )
    from policy_adapters.lerobotDP_train.train_lerobot_dp_minimal import (  # type: ignore
        _add_lerobot_to_path,
        _install_lerobot_namespace_shims,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline rollout-style LeRobot DiffusionPolicy check: feed dataset observations one by one, "
            "let the policy use its n_obs_steps/action queue, and compare sampled raw actions with GT."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to DP checkpoint_last.pt")
    parser.add_argument("--dataset-root", required=True, help="Local LeRobot dataset root")
    parser.add_argument("--repo-id", default="local/fr3_zed_lerobot_tashan_v3_train56")
    parser.add_argument("--lerobot-src", default=_guess_lerobot_src())
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--seed", type=int, default=1000, help="Diffusion sampling seed for reproducibility")
    parser.add_argument(
        "--episode",
        type=int,
        action="append",
        default=None,
        help="Episode index to evaluate. Repeat this flag for multiple episodes.",
    )
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="Save pred-vs-GT PNG curves.")
    parser.add_argument("--no-csv", action="store_true", help="Skip per-frame CSV export.")
    args = parser.parse_args()

    _set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else checkpoint_path.parent / "offline_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    _add_lerobot_to_path(args.lerobot_src)
    _install_lerobot_namespace_shims(Path(args.lerobot_src).expanduser().resolve())

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    runtime = LeRobotDPRuntime.from_checkpoint(
        checkpoint_path,
        lerobot_src=args.lerobot_src,
        device=args.device,
    )
    fps = float(_read_info(dataset_root)["fps"])
    episode_indices = _resolve_episodes(dataset_root, args.episode, args.max_episodes)

    results: list[dict[str, Any]] = []
    for ep_idx in episode_indices:
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=dataset_root,
            episodes=[ep_idx],
            delta_timestamps={ACTION_KEY: [0.0]},
            video_backend=args.video_backend,
        )
        ep_dir = output_dir / f"episode_{ep_idx:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        result = _evaluate_episode(
            runtime=runtime,
            dataset=dataset,
            episode_index=ep_idx,
            output_dir=ep_dir,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            fps=fps,
            save_csv=not args.no_csv,
            save_plot=args.plot,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    summary = _summarize(results)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "dataset_root": str(dataset_root),
            "episodes": episode_indices,
            "output_dir": str(output_dir),
            "normalization": runtime.normalization,
            "n_obs_steps": int(runtime.adapter.cfg.n_obs_steps),
            "horizon": int(runtime.adapter.cfg.horizon),
            "n_action_steps": int(runtime.adapter.cfg.n_action_steps),
            "seed": int(args.seed),
        }
    )
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _evaluate_episode(
    *,
    runtime: LeRobotDPRuntime,
    dataset: Any,
    episode_index: int,
    output_dir: Path,
    start_frame: int,
    max_frames: int,
    fps: float,
    save_csv: bool,
    save_plot: bool,
) -> dict[str, Any]:
    runtime.reset()
    start = max(0, int(start_frame))
    end = len(dataset) if max_frames <= 0 else min(len(dataset), start + int(max_frames))
    if start >= end:
        raise ValueError(f"episode {episode_index}: empty frame range start={start} end={end}")

    rows: list[dict[str, Any]] = []
    pred_actions: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []
    norm_actions: list[np.ndarray] = []

    for rel_idx in range(start, end):
        sample = dataset[rel_idx]
        canonical_obs = _sample_to_canonical_obs(sample)
        out = runtime.predict(canonical_obs)

        pred = np.asarray(out.action_raw, dtype=np.float32).reshape(7)
        norm = np.asarray(out.action_normalized, dtype=np.float32).reshape(7)
        gt = _first_action(sample[ACTION_KEY])

        pred_actions.append(pred)
        gt_actions.append(gt)
        norm_actions.append(norm)
        rows.append(_row_for_frame(rel_idx, pred, gt, norm))

    pred_arr = np.stack(pred_actions, axis=0)
    gt_arr = np.stack(gt_actions, axis=0)
    norm_arr = np.stack(norm_actions, axis=0)
    err = pred_arr - gt_arr
    abs_err = np.abs(err)
    metrics = _metrics_for_arrays(pred_arr, gt_arr, norm_arr, fps=fps)
    metrics.update(
        {
            "episode_index": int(episode_index),
            "frames": int(pred_arr.shape[0]),
            "frame_start": int(start),
            "frame_end_exclusive": int(end),
            "mean_abs_l1_all_dims": float(abs_err.mean()),
            "rmse_all_dims": float(np.sqrt(np.mean(err * err))),
        }
    )

    _write_json(output_dir / "metrics.json", metrics)
    if save_csv:
        _write_csv(output_dir / "pred_vs_gt.csv", rows)
    if save_plot:
        _save_plot(output_dir / "pred_vs_gt.png", pred_arr, gt_arr, err, episode_index)
    return metrics


def _sample_to_canonical_obs(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        FRONT_IMAGE_KEY: _image_to_hwc_uint8(sample[FRONT_IMAGE_KEY]),
        WRIST_IMAGE_KEY: _image_to_hwc_uint8(sample[WRIST_IMAGE_KEY]),
        STATE_KEY: _to_numpy(sample[STATE_KEY]).astype(np.float32).reshape(7),
        TACTILE_LEFT_KEY: np.zeros((1,), dtype=np.float32),
        TACTILE_RIGHT_KEY: np.zeros((1,), dtype=np.float32),
    }


def _image_to_hwc_uint8(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected image with 3 dims, got shape={arr.shape}")
    if arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected RGB image, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _first_action(value: Any) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32)
    if arr.ndim == 1:
        return arr.reshape(7)
    return arr.reshape(-1, 7)[0]


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _row_for_frame(frame_index: int, pred: np.ndarray, gt: np.ndarray, norm: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {"frame_index": int(frame_index)}
    err = pred - gt
    for i, name in enumerate(ACTION_NAMES):
        row[f"pred_{name}"] = float(pred[i])
        row[f"gt_{name}"] = float(gt[i])
        row[f"err_{name}"] = float(err[i])
        row[f"abs_err_{name}"] = float(abs(err[i]))
        row[f"norm_pred_{name}"] = float(norm[i])
    row["abs_err_mean"] = float(np.abs(err).mean())
    return row


def _metrics_for_arrays(pred: np.ndarray, gt: np.ndarray, norm: np.ndarray, *, fps: float) -> dict[str, Any]:
    err = pred - gt
    abs_err = np.abs(err)
    out: dict[str, Any] = {
        "fps": float(fps),
        "pred_action_max_abs": float(np.max(np.abs(pred))),
        "pred_action_norm_max_abs": float(np.max(np.abs(norm))),
    }
    if pred.shape[0] >= 2:
        step_delta = np.diff(pred, axis=0)
        out["pred_step_delta_mean_l2"] = float(np.linalg.norm(step_delta[:, :6], axis=1).mean())
        out["pred_step_delta_max_l2"] = float(np.linalg.norm(step_delta[:, :6], axis=1).max())
        out["pred_gripper_step_delta_mean_abs"] = float(np.abs(step_delta[:, 6]).mean())
        out["pred_gripper_step_delta_max_abs"] = float(np.abs(step_delta[:, 6]).max())
    for i, name in enumerate(ACTION_NAMES):
        out[f"mae_{name}"] = float(abs_err[:, i].mean())
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(err[:, i] * err[:, i])))
        out[f"bias_{name}"] = float(err[:, i].mean())
        out[f"pred_mean_{name}"] = float(pred[:, i].mean())
        out[f"gt_mean_{name}"] = float(gt[:, i].mean())
    return out


def _save_plot(path: Path, pred: np.ndarray, gt: np.ndarray, err: np.ndarray, episode_index: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    axes_flat = axes.reshape(-1)
    x = np.arange(pred.shape[0])
    for i, name in enumerate(ACTION_NAMES):
        ax = axes_flat[i]
        ax.plot(x, gt[:, i], label="gt", linewidth=1.2)
        ax.plot(x, pred[:, i], label="pred", linewidth=1.2, alpha=0.85)
        ax.plot(x, err[:, i], label="err", linewidth=0.8, alpha=0.45)
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"LeRobot DP offline rollout check - episode {episode_index}")
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"ok": False, "reason": "no episode evaluated"}
    keys = [
        "mean_abs_l1_all_dims",
        "rmse_all_dims",
        "pred_step_delta_mean_l2",
        "pred_step_delta_max_l2",
        "pred_gripper_step_delta_mean_abs",
        "pred_gripper_step_delta_max_abs",
    ]
    out: dict[str, Any] = {"ok": True, "num_episodes": len(results), "episodes_detail": results}
    for key in keys:
        vals = [float(item[key]) for item in results if key in item and math.isfinite(float(item[key]))]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_max"] = float(np.max(vals))
    return out


def _resolve_episodes(dataset_root: Path, requested: list[int] | None, max_episodes: int) -> list[int]:
    if requested:
        return [int(x) for x in requested]
    info = _read_info(dataset_root)
    total = int(info.get("total_episodes", 0))
    if total <= 0:
        raise ValueError(f"could not resolve total_episodes from {dataset_root / 'meta' / 'info.json'}")
    return list(range(min(total, max(1, int(max_episodes)))))


def _read_info(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "meta" / "info.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _guess_lerobot_src() -> str:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "lerobot" / "src"
        if candidate.exists():
            return str(candidate)
    return str(Path.cwd() / "lerobot" / "src")


if __name__ == "__main__":
    main()

