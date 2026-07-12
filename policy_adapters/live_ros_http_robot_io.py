from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - deployment machines normally have cv2.
    cv2 = None  # type: ignore[assignment]

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except ModuleNotFoundError:  # pragma: no cover - lets this file be imported off the robot.
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    HistoryPolicy = QoSProfile = ReliabilityPolicy = None  # type: ignore[assignment]
    Image = String = None  # type: ignore[assignment]

try:
    from .utils.common import (
        FRONT_IMAGE_KEY,
        STATE_KEY,
        TACTILE_LEFT_KEY,
        TACTILE_RIGHT_KEY,
        WRIST_IMAGE_KEY,
        AdapterImageConfig,
        ActionTarget,
        parse_image_roi_config,
        preprocess_rgb,
    )
    from .utils.action_space import pose7_to_state7
    from .utils.online_policy_adapter import assert_canonical_observation
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from policy_adapters.utils.common import (  # type: ignore
        FRONT_IMAGE_KEY,
        STATE_KEY,
        TACTILE_LEFT_KEY,
        TACTILE_RIGHT_KEY,
        WRIST_IMAGE_KEY,
        AdapterImageConfig,
        ActionTarget,
        parse_image_roi_config,
        preprocess_rgb,
    )
    from policy_adapters.utils.action_space import pose7_to_state7  # type: ignore
    from policy_adapters.utils.online_policy_adapter import (  # type: ignore
        assert_canonical_observation,
    )


CURRENT_GRIPPER_WIDTH_M_KEY = "robot.current_gripper_width_m"


@dataclass(frozen=True)
class LiveRosHttpRobotIOConfig:
    server_url: str = os.environ.get("FRANKA_SERVER_URL", "http://xxx:5000")
    http_timeout_sec: float = 0.15

    front_image_topic: str = "/zed/zed_node/rgb/color/rect/image"
    wrist_image_topic: str = "/camera/d405/color/image_raw"
    robot_state_topic: str = "/hilserl/robot_state_json"
    tactile_left_topic: str = "/tashan_tactile/left/data"
    tactile_right_topic: str = "/tashan_tactile/right/data"
    front_roi: Any | None = None
    front_roi_size: int = 256

    max_image_dt_sec: float = 0.10
    max_wrist_image_dt_sec: float = 0.10
    max_state_dt_sec: float = 0.10
    max_tactile_dt_sec: float = 0.10

    observation_timeout_sec: float = 5.0
    spin_once_timeout_sec: float = 0.02
    require_tactile: bool = False
    tactile_zero_len: int = 1
    send_gripper: bool = True
    ros_node_name: str = "policy_live_ros_http_io"


@dataclass
class TimedItem:
    t: float
    payload: Any


class TimedBuffer:
    def __init__(self, maxlen: int = 512):
        self._buf: Deque[TimedItem] = deque(maxlen=maxlen)

    def append(self, t: float, payload: Any) -> None:
        self._buf.append(TimedItem(float(t), payload))

    def latest(self) -> Optional[TimedItem]:
        if not self._buf:
            return None
        return self._buf[-1]

    def nearest(self, t: float) -> Optional[TimedItem]:
        if not self._buf:
            return None
        return min(self._buf, key=lambda item: abs(item.t - t))


class _LiveObservationNode(Node):  # type: ignore[misc,valid-type]
    def __init__(self, cfg: LiveRosHttpRobotIOConfig):
        if rclpy is None or Image is None or String is None:
            raise RuntimeError(
                "ROS2 Python modules are not available. Run this on the robot/perception machine "
                "after sourcing the ROS2 workspace."
            )
        super().__init__(cfg.ros_node_name)
        self.cfg = cfg
        self.front_image_buffer = TimedBuffer(maxlen=512)
        self.wrist_image_buffer = TimedBuffer(maxlen=512)
        self.state_buffer = TimedBuffer(maxlen=4096)
        self.tactile_left_buffer = TimedBuffer(maxlen=4096)
        self.tactile_right_buffer = TimedBuffer(maxlen=4096)
        self._tactile_available = False

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        log_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )
        fast_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image, cfg.front_image_topic, self._front_image_cb, image_qos)
        self.create_subscription(Image, cfg.wrist_image_topic, self._wrist_image_cb, image_qos)
        self.create_subscription(String, cfg.robot_state_topic, self._state_cb, log_qos)

        tactile_type = _resolve_tactile_msg_type()
        if tactile_type is not None:
            self.create_subscription(tactile_type, cfg.tactile_left_topic, self._tactile_left_cb, fast_qos)
            self.create_subscription(tactile_type, cfg.tactile_right_topic, self._tactile_right_cb, fast_qos)
            self._tactile_available = True
        elif cfg.require_tactile:
            raise RuntimeError("tashan_tactile.msg.TactileFrame is required but could not be imported")

        self.get_logger().info(f"subscribe front image: {cfg.front_image_topic}")
        self.get_logger().info(f"subscribe wrist image: {cfg.wrist_image_topic}")
        self.get_logger().info(f"subscribe robot state: {cfg.robot_state_topic}")
        if self._tactile_available:
            self.get_logger().info(f"subscribe tactile left: {cfg.tactile_left_topic}")
            self.get_logger().info(f"subscribe tactile right: {cfg.tactile_right_topic}")
        else:
            self.get_logger().warn("tactile message package unavailable; using zero tactile placeholders")

    def _front_image_cb(self, msg: Any) -> None:
        self.front_image_buffer.append(time.monotonic(), msg)

    def _wrist_image_cb(self, msg: Any) -> None:
        self.wrist_image_buffer.append(time.monotonic(), msg)

    def _state_cb(self, msg: Any) -> None:
        event = _parse_json_str(str(msg.data))
        t = float(event.get("t_query_mid", event.get("publish_time", time.monotonic())))
        self.state_buffer.append(t, event)

    def _tactile_left_cb(self, msg: Any) -> None:
        self.tactile_left_buffer.append(time.monotonic(), _tactile_msg_to_array(msg))

    def _tactile_right_cb(self, msg: Any) -> None:
        self.tactile_right_buffer.append(time.monotonic(), _tactile_msg_to_array(msg))

    def nearest_bundle(self, t_frame: float) -> dict[str, Optional[TimedItem]]:
        return {
            "front_image": self.front_image_buffer.nearest(t_frame),
            "wrist_image": self.wrist_image_buffer.nearest(t_frame),
            "state": self.state_buffer.nearest(t_frame),
            "tactile_left": self.tactile_left_buffer.nearest(t_frame),
            "tactile_right": self.tactile_right_buffer.nearest(t_frame),
        }


class _FrankaHttpClient:
    def __init__(self, server_url: str, timeout_sec: float):
        try:
            import requests
        except ModuleNotFoundError as exc:  # pragma: no cover - deployment dependency.
            raise RuntimeError("requests is required for HTTP robot commands") from exc
        self._requests = requests
        self.server_url = str(server_url).rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.session = requests.Session()

    def post(self, route: str, payload: dict[str, Any] | None = None, *, timeout_sec: float | None = None) -> Any:
        route = route if route.startswith("/") else f"/{route}"
        timeout = self.timeout_sec if timeout_sec is None else float(timeout_sec)
        if payload is None:
            response = self.session.post(f"{self.server_url}{route}", timeout=timeout)
        else:
            response = self.session.post(f"{self.server_url}{route}", json=payload, timeout=timeout)
        response.raise_for_status()
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return response.text

    def command_pose(self, pose_xyz_xyzw: np.ndarray) -> None:
        pose = np.asarray(pose_xyz_xyzw, dtype=float).reshape(7)
        self.post("/pose", {"arr": pose.tolist()})

    def move_gripper_width(self, gripper_width_m: float) -> None:
        self.post("/move_gripper", {"gripper_width": float(gripper_width_m)}, timeout_sec=0.5)

    def close(self) -> None:
        self.session.close()


class LiveRosHttpRobotIO:
    """RobotIO implementation for live ROS observations and HIL-SERL HTTP commands."""

    def __init__(self, cfg: LiveRosHttpRobotIOConfig | None = None):
        self.cfg = cfg or LiveRosHttpRobotIOConfig()
        if rclpy is None:
            raise RuntimeError(
                "rclpy is not available. This live RobotIO must run on the ROS2 deployment machine."
            )
        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self.node = _LiveObservationNode(self.cfg)
        self.client = _FrankaHttpClient(self.cfg.server_url, self.cfg.http_timeout_sec)
        self._front_roi = parse_image_roi_config(self.cfg.front_roi, name="front_roi")
        front_roi_size = int(self.cfg.front_roi_size)
        if front_roi_size <= 0:
            raise ValueError(f"front_roi_size must be positive, got {front_roi_size}")
        self._front_roi_image = AdapterImageConfig(height=front_roi_size, width=front_roi_size)
        self.last_sync: dict[str, Any] = {}

    def get_observation(self) -> dict[str, Any]:
        deadline = time.monotonic() + float(self.cfg.observation_timeout_sec)
        last_reason = "no data yet"
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=float(self.cfg.spin_once_timeout_sec))
            t_frame = time.monotonic()
            obs, sync, reason = self._try_build_observation(t_frame)
            if obs is not None:
                self.last_sync = sync
                return obs
            last_reason = reason
        raise TimeoutError(
            f"could not build a synchronized policy observation within "
            f"{self.cfg.observation_timeout_sec:.2f}s: {last_reason}"
        )

    def send_target(self, target: ActionTarget) -> None:
        self.client.command_pose(target.target_pose7)
        if self.cfg.send_gripper:
            self.client.move_gripper_width(float(target.target_gripper_width))

    def stop(self) -> None:
        try:
            self.client.close()
        finally:
            try:
                self.node.destroy_node()
            finally:
                if self._owns_rclpy and rclpy is not None and rclpy.ok():
                    rclpy.shutdown()

    def _try_build_observation(
        self,
        t_frame: float,
    ) -> tuple[dict[str, Any] | None, dict[str, Any], str]:
        bundle = self.node.nearest_bundle(t_frame)
        ok_front, dt_front, reason_front = _valid_pair(
            "front_image", bundle["front_image"], t_frame, self.cfg.max_image_dt_sec
        )
        ok_wrist, dt_wrist, reason_wrist = _valid_pair(
            "wrist_image", bundle["wrist_image"], t_frame, self.cfg.max_wrist_image_dt_sec
        )
        ok_state, dt_state, reason_state = _valid_pair(
            "state", bundle["state"], t_frame, self.cfg.max_state_dt_sec
        )
        ok_left, dt_left, reason_left = _valid_pair(
            "tactile_left", bundle["tactile_left"], t_frame, self.cfg.max_tactile_dt_sec
        )
        ok_right, dt_right, reason_right = _valid_pair(
            "tactile_right", bundle["tactile_right"], t_frame, self.cfg.max_tactile_dt_sec
        )

        if not (ok_front and ok_wrist and ok_state):
            return (
                None,
                {},
                f"front={reason_front} wrist={reason_wrist} state={reason_state}",
            )
        if self.cfg.require_tactile and not (ok_left and ok_right):
            return (
                None,
                {},
                f"tactile_left={reason_left} tactile_right={reason_right}",
            )

        assert bundle["front_image"] is not None
        assert bundle["wrist_image"] is not None
        assert bundle["state"] is not None
        try:
            tactile_left = (
                np.asarray(bundle["tactile_left"].payload, dtype=np.float32).reshape(-1)
                if ok_left and bundle["tactile_left"] is not None
                else np.zeros((int(self.cfg.tactile_zero_len),), dtype=np.float32)
            )
            tactile_right = (
                np.asarray(bundle["tactile_right"].payload, dtype=np.float32).reshape(-1)
                if ok_right and bundle["tactile_right"] is not None
                else np.zeros((int(self.cfg.tactile_zero_len),), dtype=np.float32)
            )
            state_event = bundle["state"].payload
            front_rgb = image_msg_to_rgb8(bundle["front_image"].payload)
            if self._front_roi is not None:
                front_rgb = preprocess_rgb(
                    front_rgb,
                    self._front_roi_image,
                    roi=self._front_roi,
                    name=FRONT_IMAGE_KEY,
                )
            obs = {
                FRONT_IMAGE_KEY: front_rgb,
                WRIST_IMAGE_KEY: image_msg_to_rgb8(bundle["wrist_image"].payload),
                STATE_KEY: state_event_to_state7(state_event),
                TACTILE_LEFT_KEY: tactile_left,
                TACTILE_RIGHT_KEY: tactile_right,
            }
            gripper_width_m = state_event_to_gripper_width_m(state_event)
            if gripper_width_m is not None:
                obs[CURRENT_GRIPPER_WIDTH_M_KEY] = np.float32(gripper_width_m)
            assert_canonical_observation(obs)
        except Exception as exc:
            return None, {}, f"observation conversion failed: {exc!r}"

        sync = {
            "dt_front_image": dt_front,
            "dt_wrist_image": dt_wrist,
            "dt_state": dt_state,
            "dt_tactile_left": dt_left if ok_left else None,
            "dt_tactile_right": dt_right if ok_right else None,
            "reason_tactile_left": reason_left,
            "reason_tactile_right": reason_right,
        }
        return obs, sync, "ok"


def create_robot_io(**kwargs: Any) -> LiveRosHttpRobotIO:
    return LiveRosHttpRobotIO(LiveRosHttpRobotIOConfig(**kwargs))


def image_msg_to_rgb8(msg: Any) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("cv2 is required to convert ROS images")
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    encoding = str(msg.encoding).lower()
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")

    raw = _uint8_message_data(msg.data)
    if encoding in ("rgb8", "bgr8"):
        row_bytes = width * 3
        arr = raw.reshape(height, step)[:, :row_bytes].reshape(height, width, 3)
        if encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(arr)
    if encoding in ("rgba8", "bgra8"):
        row_bytes = width * 4
        arr = raw.reshape(height, step)[:, :row_bytes].reshape(height, width, 4)
        code = cv2.COLOR_RGBA2RGB if encoding == "rgba8" else cv2.COLOR_BGRA2RGB
        return np.ascontiguousarray(cv2.cvtColor(arr, code))
    if encoding in ("mono8", "8uc1"):
        arr = raw.reshape(height, step)[:, :width].reshape(height, width)
        return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def state_event_to_state7(state_event: dict[str, Any]) -> np.ndarray:
    state = _state_payload(state_event)
    pose = _first_present(state, ("pose", "ee_pose", "pos"))
    if pose is None:
        pose = _first_present(state_event, ("pose", "ee_pose", "pos"))
    gripper = _first_present(state, ("gripper_pos", "gripper", "gripper_width"))
    if gripper is None:
        gripper = _first_present(state_event, ("gripper_pos", "gripper", "gripper_width"))
    if pose is None or gripper is None:
        raise ValueError("robot state is missing ee pose or gripper width")
    pose7 = np.asarray(pose, dtype=np.float32).reshape(-1)[:7]
    if pose7.shape != (7,):
        raise ValueError(f"robot pose must have 7 values, got {pose7.shape}")
    gripper_width = float(np.asarray(gripper, dtype=np.float32).reshape(-1)[0])
    return pose7_to_state7(pose7, gripper_width).astype(np.float32)


def state_event_to_gripper_width_m(state_event: dict[str, Any]) -> float | None:
    state = _state_payload(state_event)
    value = _first_present(state, ("gripper_width", "width_m"))
    if value is None:
        value = _first_present(state_event, ("gripper_width", "width_m"))
    if value is None:
        maybe_gripper = _first_present(state, ("gripper",))
        if maybe_gripper is not None:
            maybe = float(np.asarray(maybe_gripper, dtype=np.float32).reshape(-1)[0])
            if 0.0 <= maybe <= 0.20:
                return maybe
        return None
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def _state_payload(state_event: dict[str, Any]) -> dict[str, Any]:
    state = state_event.get("state", state_event)
    if not isinstance(state, dict):
        raise ValueError("robot state JSON does not contain a state dict")
    return state


def _resolve_tactile_msg_type() -> Any | None:
    try:
        from tashan_tactile.msg import TactileFrame

        return TactileFrame
    except ModuleNotFoundError:
        return None


def _tactile_msg_to_array(msg: Any) -> np.ndarray:
    return np.asarray(list(msg.data), dtype=np.float32).reshape(-1)


def _parse_json_str(value: str) -> dict[str, Any]:
    try:
        out = json.loads(value)
        return out if isinstance(out, dict) else {"value": out}
    except Exception as exc:
        return {"parse_error": repr(exc), "raw": value}


def _valid_pair(
    name: str,
    item: Optional[TimedItem],
    t_frame: float,
    max_dt_sec: float,
) -> tuple[bool, float | None, str]:
    if item is None:
        return False, None, f"missing_{name}"
    dt = float(item.t - t_frame)
    if abs(dt) > float(max_dt_sec):
        return False, dt, f"stale_{name}_dt_{dt:.3f}"
    return True, dt, "ok"


def _uint8_message_data(data: Any) -> np.ndarray:
    try:
        return np.frombuffer(data, dtype=np.uint8)
    except TypeError:
        return np.asarray(data, dtype=np.uint8)


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None
