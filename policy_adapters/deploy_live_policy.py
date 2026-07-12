from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    from .utils.common import STATE_KEY, ActionTarget
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from policy_adapters.utils.common import STATE_KEY, ActionTarget  # type: ignore


CURRENT_GRIPPER_WIDTH_M_KEY = "robot.current_gripper_width_m"


class RobotIO(Protocol):
    def get_observation(self) -> dict[str, Any]:
        """Return a canonical policy observation dict."""

    def send_target(self, target: ActionTarget) -> None:
        """Send a target pose/gripper command."""

    def stop(self) -> None:
        """Best-effort cleanup."""


class Runtime(Protocol):
    adapter: Any

    def predict(self, canonical_obs: dict[str, Any]) -> Any:
        """Return an object with action_raw and target."""

    def reset(self) -> None:
        """Clear runtime queues/history."""


@dataclass(frozen=True)
class SafetyLimits:
    max_delta_xyz_m: float = 0.03
    max_delta_rotvec_rad: float = 0.08
    max_gripper_step_m: float = 0.02
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    gripper_min_m: float | None = None
    gripper_max_m: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Live deployment loop for LeRobot ACT or LeRobot DiffusionPolicy. "
            "It reads canonical observations from a RobotIO module, runs policy inference, applies "
            "low-speed safety clipping, sends ActionTarget commands, and logs every step."
        )
    )
    parser.add_argument("--policy-kind", choices=("act", "lerobot-dp"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--io-module",
        default=_default_live_io_module(),
        help="Python file exporting create_robot_io(**kwargs). Defaults to live_ros_http_robot_io.py.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--speed-scale", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lerobot-src", default=_guess_lerobot_src())

    parser.add_argument("--server-url", default=None)
    parser.add_argument("--http-timeout-sec", type=float, default=None)
    parser.add_argument("--front-image-topic", default=None)
    parser.add_argument("--wrist-image-topic", default=None)
    parser.add_argument("--robot-state-topic", default=None)
    parser.add_argument("--tactile-left-topic", default=None)
    parser.add_argument("--tactile-right-topic", default=None)
    parser.add_argument(
        "--front-roi",
        default=None,
        help=(
            "Optional fixed front-camera ROI before live model input, formatted as x1,y1,x2,y2. "
            "When set, RobotIO crops front then resizes it to --front-roi-size square pixels."
        ),
    )
    parser.add_argument("--front-roi-size", type=int, default=256)
    parser.add_argument("--require-tactile", action="store_true")
    parser.add_argument("--no-send-gripper", action="store_true")
    parser.add_argument("--observation-timeout-sec", type=float, default=None)
    parser.add_argument("--max-image-dt-sec", type=float, default=None)
    parser.add_argument("--max-wrist-image-dt-sec", type=float, default=None)
    parser.add_argument("--max-state-dt-sec", type=float, default=None)
    parser.add_argument("--max-tactile-dt-sec", type=float, default=None)

    parser.add_argument("--max-delta-xyz-m", type=float, default=0.01)
    parser.add_argument("--max-delta-rotvec-rad", type=float, default=0.08)
    parser.add_argument("--max-gripper-step-m", type=float, default=0.005)
    parser.add_argument("--workspace-min", default=None, help="Optional x,y,z lower bounds, e.g. 0.25,-0.45,0.02")
    parser.add_argument("--workspace-max", default=None, help="Optional x,y,z upper bounds, e.g. 0.85,0.45,0.65")
    parser.add_argument("--gripper-min-m", type=float, default=None)
    parser.add_argument("--gripper-max-m", type=float, default=None)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    if args.rate_hz <= 0:
        raise ValueError(f"--rate-hz must be positive, got {args.rate_hz}")
    if not (0.0 < args.speed_scale <= 2.0):
        raise ValueError(f"--speed-scale must be in (0, 1], got {args.speed_scale}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "deploy_config.json", vars(args))

    runtime = _load_runtime(args)
    runtime.reset()
    robot = _load_robot_io(args.io_module, _io_kwargs(args))
    limits = SafetyLimits(
        max_delta_xyz_m=float(args.max_delta_xyz_m),
        max_delta_rotvec_rad=float(args.max_delta_rotvec_rad),
        max_gripper_step_m=float(args.max_gripper_step_m),
        workspace_min=_parse_vec3(args.workspace_min, "--workspace-min"),
        workspace_max=_parse_vec3(args.workspace_max, "--workspace-max"),
        gripper_min_m=args.gripper_min_m,
        gripper_max_m=args.gripper_max_m,
    )
    _write_json(output_dir / "safety_limits.json", asdict(limits))

    rows: list[dict[str, Any]] = []
    dt = 1.0 / float(args.rate_hz)
    # 实时推理
    try:
        for step in range(int(args.steps)):
            step_start = time.perf_counter()
            try:
                obs = robot.get_observation()
            # 获取观测超时
            except TimeoutError as exc:
                after_obs = time.perf_counter()
                elapsed_before_sleep = after_obs - step_start
                sleep_s = max(0.0, dt - elapsed_before_sleep)
                row = _skipped_row(
                    step=step,
                    reason="observation_timeout",
                    error=str(exc),
                    step_s=elapsed_before_sleep,
                    obs_s=elapsed_before_sleep,
                    sleep_s=sleep_s,
                    overrun_s=max(0.0, elapsed_before_sleep - dt),
                )
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue
            after_obs = time.perf_counter()
            raw_state7 = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(7)
            out = runtime.predict(obs)
            after_predict = time.perf_counter()
            model_action = np.asarray(out.action_raw, dtype=np.float32).reshape(7)
            current_gripper_width_m = _current_gripper_width_m(obs, raw_state7)
            safe_action, action_clipped = _safe_scaled_action(
                current_state7=raw_state7,
                action7=model_action,
                speed_scale=float(args.speed_scale),
                limits=limits,
                current_gripper_width_m=current_gripper_width_m,
            )
            safe_target = runtime.adapter.action_to_target(raw_state7, safe_action)
            safe_target, target_clipped = _clip_target(safe_target, limits)

            sent = step >= int(args.warmup_steps) and not args.dry_run
            if sent:
                robot.send_target(safe_target)
            after_send = time.perf_counter()

            elapsed_before_sleep = after_send - step_start
            sleep_s = max(0.0, dt - elapsed_before_sleep)
            overrun_s = max(0.0, elapsed_before_sleep - dt)

            row = _rollout_row(
                step=step,
                raw_state7=raw_state7,
                model_action=model_action,
                safe_action=safe_action,
                model_target=out.target,
                safe_target=safe_target,
                clipped=bool(action_clipped or target_clipped),
                sent=sent,
                step_s=elapsed_before_sleep,
                obs_s=after_obs - step_start,
                predict_s=after_predict - after_obs,
                send_s=after_send - after_predict,
                sleep_s=sleep_s,
                overrun_s=overrun_s,
                sync=getattr(robot, "last_sync", None),
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        try:
            robot.stop()
        finally:
            _write_csv(output_dir / "deploy_log.csv", rows)
            _write_json(output_dir / "summary.json", _summarize_rows(rows))


def _load_runtime(args: argparse.Namespace) -> Runtime:
    if args.policy_kind == "act":
        try:
            from .lerobotACT_train.deploy_lerobot_act_runtime import LeRobotACTRuntime
        except ImportError:
            from policy_adapters.lerobotACT_train.deploy_lerobot_act_runtime import (  # type: ignore
                LeRobotACTRuntime,
            )

        return LeRobotACTRuntime.from_checkpoint(
            args.checkpoint,
            lerobot_src=args.lerobot_src,
            device=args.device,
        )
    if args.policy_kind == "lerobot-dp":
        try:
            from .lerobotDP_train.deploy_lerobot_dp_runtime import LeRobotDPRuntime
        except ImportError:
            from policy_adapters.lerobotDP_train.deploy_lerobot_dp_runtime import (  # type: ignore
                LeRobotDPRuntime,
            )

        return LeRobotDPRuntime.from_checkpoint(
            args.checkpoint,
            lerobot_src=args.lerobot_src,
            device=args.device,
        )
    raise ValueError(f"unsupported policy kind: {args.policy_kind}")


def _load_robot_io(path: str, kwargs: dict[str, Any]) -> RobotIO:
    module_path = Path(path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("policy_live_robot_io", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load RobotIO module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "create_robot_io"):
        raise AttributeError(f"{module_path} must export create_robot_io(**kwargs)")
    return module.create_robot_io(**kwargs)


def _io_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "server_url": args.server_url,
        "http_timeout_sec": args.http_timeout_sec,
        "front_image_topic": args.front_image_topic,
        "wrist_image_topic": args.wrist_image_topic,
        "robot_state_topic": args.robot_state_topic,
        "tactile_left_topic": args.tactile_left_topic,
        "tactile_right_topic": args.tactile_right_topic,
        "observation_timeout_sec": args.observation_timeout_sec,
        "max_image_dt_sec": args.max_image_dt_sec,
        "max_wrist_image_dt_sec": args.max_wrist_image_dt_sec,
        "max_state_dt_sec": args.max_state_dt_sec,
        "max_tactile_dt_sec": args.max_tactile_dt_sec,
        "require_tactile": bool(args.require_tactile),
        "send_gripper": not bool(args.no_send_gripper),
    }
    if args.front_roi is not None:
        mapping["front_roi"] = args.front_roi
        mapping["front_roi_size"] = args.front_roi_size
    return {key: value for key, value in mapping.items() if value is not None}


def _safe_scaled_action(
    *,
    current_state7: np.ndarray,
    action7: np.ndarray,
    speed_scale: float,
    limits: SafetyLimits,
    current_gripper_width_m: float | None = None,
) -> tuple[np.ndarray, bool]:
    action = np.asarray(action7, dtype=np.float32).reshape(7).copy()
    original = action.copy()
    action[:6] *= float(speed_scale)

    xyz_norm = float(np.linalg.norm(action[:3]))
    if xyz_norm > limits.max_delta_xyz_m:
        action[:3] *= limits.max_delta_xyz_m / max(xyz_norm, 1e-8)

    rot_norm = float(np.linalg.norm(action[3:6]))
    if rot_norm > limits.max_delta_rotvec_rad:
        action[3:6] *= limits.max_delta_rotvec_rad / max(rot_norm, 1e-8)

    current_gripper = (
        float(current_gripper_width_m)
        if current_gripper_width_m is not None
        else float(np.asarray(current_state7, dtype=np.float32).reshape(7)[6])
    )
    gripper_delta = float(action[6] - current_gripper)
    gripper_delta = float(np.clip(gripper_delta, -limits.max_gripper_step_m, limits.max_gripper_step_m))
    action[6] = np.float32(current_gripper + gripper_delta)

    clipped = bool(np.any(np.abs(action - original) > 1e-8))
    return action.astype(np.float32), clipped


def _current_gripper_width_m(obs: dict[str, Any], state7: np.ndarray) -> float | None:
    if CURRENT_GRIPPER_WIDTH_M_KEY in obs:
        value = float(np.asarray(obs[CURRENT_GRIPPER_WIDTH_M_KEY], dtype=np.float32).reshape(-1)[0])
        if np.isfinite(value):
            return value
    fallback = float(np.asarray(state7, dtype=np.float32).reshape(7)[6])
    if 0.0 <= fallback <= 0.20:
        return fallback
    return None


def _clip_target(target: ActionTarget, limits: SafetyLimits) -> tuple[ActionTarget, bool]:
    pose = np.asarray(target.target_pose7, dtype=np.float32).reshape(7).copy()
    gripper = float(target.target_gripper_width)
    original_pose = pose.copy()
    original_gripper = gripper
    if limits.workspace_min is not None or limits.workspace_max is not None:
        low = np.asarray(
            limits.workspace_min if limits.workspace_min is not None else (-np.inf, -np.inf, -np.inf),
            dtype=np.float32,
        )
        high = np.asarray(
            limits.workspace_max if limits.workspace_max is not None else (np.inf, np.inf, np.inf),
            dtype=np.float32,
        )
        pose[:3] = np.clip(pose[:3], low, high)
    if limits.gripper_min_m is not None or limits.gripper_max_m is not None:
        gripper = float(
            np.clip(
                gripper,
                -np.inf if limits.gripper_min_m is None else limits.gripper_min_m,
                np.inf if limits.gripper_max_m is None else limits.gripper_max_m,
            )
        )
    clipped = bool(np.any(np.abs(pose - original_pose) > 1e-8) or abs(gripper - original_gripper) > 1e-8)
    return ActionTarget(target_pose7=pose, target_gripper_width=gripper), clipped


def _rollout_row(
    *,
    step: int,
    raw_state7: np.ndarray,
    model_action: np.ndarray,
    safe_action: np.ndarray,
    model_target: ActionTarget,
    safe_target: ActionTarget,
    clipped: bool,
    sent: bool,
    step_s: float,
    obs_s: float,
    predict_s: float,
    send_s: float,
    sleep_s: float,
    overrun_s: float,
    sync: dict[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": int(step),
        "status": "ok",
        "sent": bool(sent),
        "clipped": bool(clipped),
        "step_s": float(step_s),
        "obs_s": float(obs_s),
        "predict_s": float(predict_s),
        "send_s": float(send_s),
        "sleep_s": float(sleep_s),
        "overrun_s": float(overrun_s),
    }
    if sync:
        for key, value in sync.items():
            if isinstance(value, (int, float, bool)) or value is None:
                row[f"sync_{key}"] = value
    for i, name in enumerate(("x", "y", "z", "rotvec_x", "rotvec_y", "rotvec_z", "gripper")):
        row[f"state_{name}"] = float(raw_state7[i])
        row[f"model_action_{name}"] = float(model_action[i])
        row[f"safe_action_{name}"] = float(safe_action[i])
    for i, name in enumerate(("x", "y", "z", "qx", "qy", "qz", "qw")):
        row[f"model_target_pose_{name}"] = float(model_target.target_pose7[i])
        row[f"safe_target_pose_{name}"] = float(safe_target.target_pose7[i])
    row["model_target_gripper_width"] = float(model_target.target_gripper_width)
    row["safe_target_gripper_width"] = float(safe_target.target_gripper_width)
    return row


def _skipped_row(
    *,
    step: int,
    reason: str,
    error: str,
    step_s: float,
    obs_s: float,
    sleep_s: float,
    overrun_s: float,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "status": f"skipped_{reason}",
        "sent": False,
        "clipped": False,
        "step_s": float(step_s),
        "obs_s": float(obs_s),
        "predict_s": 0.0,
        "send_s": 0.0,
        "sleep_s": float(sleep_s),
        "overrun_s": float(overrun_s),
        "skip_reason": str(reason),
        "error": str(error),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"ok": False, "reason": "no rollout rows"}
    timing_keys = ("step_s", "obs_s", "predict_s", "send_s", "sleep_s", "overrun_s")
    timing_summary: dict[str, float] = {}
    for key in timing_keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            timing_summary[f"mean_{key}"] = float(np.mean(values))
            timing_summary[f"max_{key}"] = float(np.max(values))
    return {
        "ok": True,
        "steps": len(rows),
        "policy_steps": int(sum(1 for row in rows if row.get("status", "ok") == "ok")),
        "skipped_steps": int(sum(1 for row in rows if row.get("status", "ok") != "ok")),
        "sent_steps": int(sum(1 for row in rows if row.get("sent"))),
        "clipped_steps": int(sum(1 for row in rows if row.get("clipped"))),
        **timing_summary,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_vec3(value: str | None, name: str) -> tuple[float, float, float] | None:
    if value is None or str(value).strip() == "":
        return None
    arr = [float(x.strip()) for x in str(value).split(",")]
    if len(arr) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated floats")
    return (arr[0], arr[1], arr[2])


def _guess_lerobot_src() -> str:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "lerobot" / "src"
        if candidate.exists():
            return str(candidate)
    return str(Path.cwd() / "lerobot" / "src")


def _default_live_io_module() -> str:
    return str(Path(__file__).resolve().with_name("live_ros_http_robot_io.py"))


if __name__ == "__main__":
    main()
