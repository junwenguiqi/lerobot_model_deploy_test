from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

import cv2
import numpy as np

from .action_space import relative_action7_to_target_pose7
from .online_policy_adapter import (
    FRONT_IMAGE_KEY,
    STATE_KEY,
    TACTILE_LEFT_KEY,
    TACTILE_RIGHT_KEY,
    WRIST_IMAGE_KEY,
    assert_canonical_observation,
)


ACTION_KEY = "action"


@dataclass(frozen=True)
class AdapterImageConfig:
    height: int = 256
    width: int = 256
    interpolation: int = cv2.INTER_AREA

    @property
    def chw_shape(self) -> tuple[int, int, int]:
        return (3, int(self.height), int(self.width))

    @property
    def hwc_shape(self) -> tuple[int, int, int]:
        return (int(self.height), int(self.width), 3)


@dataclass(frozen=True)
class ImageRoiConfig:
    """Fixed image ROI in original image pixel coordinates: left, top, right, bottom."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (int(self.left), int(self.top), int(self.right), int(self.bottom))

    def validate(self, image_size: tuple[int, int], *, name: str = "roi") -> None:
        width, height = int(image_size[0]), int(image_size[1])
        left, top, right, bottom = self.box
        if left < 0 or top < 0 or right > width or bottom > height:
            raise ValueError(f"{name}={self.box} is outside image size {(width, height)}")
        if right <= left or bottom <= top:
            raise ValueError(f"{name} must satisfy right>left and bottom>top, got {self.box}")

    def as_dict(self) -> dict[str, int]:
        left, top, right, bottom = self.box
        return {"left": left, "top": top, "right": right, "bottom": bottom}


@dataclass(frozen=True)
class ActionTarget:
    target_pose7: np.ndarray
    target_gripper_width: float


@dataclass(frozen=True)
class PolicyAdapterSpec:
    name: str
    input_keys: tuple[str, ...]
    output_key: str
    state_dim: int
    action_dim: int
    image_shape_chw: tuple[int, int, int]
    action_semantics: str
    tactile_used: bool
    notes: tuple[str, ...]


def resize_rgb(image: np.ndarray, cfg: AdapterImageConfig) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got shape={image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    out = cv2.resize(
        image,
        (int(cfg.width), int(cfg.height)),
        interpolation=int(cfg.interpolation),
    )
    return np.ascontiguousarray(out)


def preprocess_rgb(
    image: np.ndarray,
    cfg: AdapterImageConfig,
    *,
    roi: ImageRoiConfig | None = None,
    name: str = "image",
) -> np.ndarray:
    image = crop_rgb(image, roi, name=name)
    return resize_rgb(image, cfg)


def crop_rgb(image: np.ndarray, roi: ImageRoiConfig | None, *, name: str = "image") -> np.ndarray:
    image = np.asarray(image)
    if roi is None:
        return image
    if image.ndim != 3:
        raise ValueError(f"expected HWC image for {name}, got shape={image.shape}")
    roi.validate((image.shape[1], image.shape[0]), name=f"{name}_roi")
    left, top, right, bottom = roi.box
    return np.ascontiguousarray(image[top:bottom, left:right, :])


def chw_float32_image(
    image: np.ndarray,
    cfg: AdapterImageConfig,
    *,
    roi: ImageRoiConfig | None = None,
    name: str = "image",
) -> np.ndarray:
    resized = preprocess_rgb(image, cfg, roi=roi, name=name)
    return np.ascontiguousarray(np.moveaxis(resized.astype(np.float32) / 255.0, -1, 0))


def parse_image_roi_config(value: Any, *, name: str = "roi") -> ImageRoiConfig | None:
    if value is None:
        return None
    if isinstance(value, ImageRoiConfig):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "null"}:
            return None
        parts = [part.strip() for part in text.split(",")]
    elif isinstance(value, dict):
        roi = ImageRoiConfig(
            left=int(value["left"]),
            top=int(value["top"]),
            right=int(value["right"]),
            bottom=int(value["bottom"]),
        )
        roi.validate((roi.right, roi.bottom), name=name)
        return roi
    else:
        parts = list(value)

    if len(parts) != 4:
        raise ValueError(f"{name} must be formatted as x1,y1,x2,y2, got {value!r}")
    try:
        left, top, right, bottom = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{name} values must be integers, got {value!r}") from exc
    roi = ImageRoiConfig(left=left, top=top, right=right, bottom=bottom)
    roi.validate((right, bottom), name=name)
    return roi


def state7(obs: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(-1)
    if arr.shape != (7,):
        raise ValueError(f"{STATE_KEY} must be float32[7], got shape={arr.shape}")
    return arr


def current_pose7_from_state7(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(7)
    return np.concatenate(
        [state[:3], _rotvec_to_quat_xyzw(state[3:6])],
        axis=0,
    ).astype(np.float32)


def compose_relative_action(
    current_state7: np.ndarray,
    action7: np.ndarray,
) -> ActionTarget:
    pose7, gripper = relative_action7_to_target_pose7(
        current_pose7=current_pose7_from_state7(current_state7),
        action7=np.asarray(action7, dtype=np.float32).reshape(7),
    )
    return ActionTarget(target_pose7=pose7, target_gripper_width=float(gripper))


class ObservationHistory:
    def __init__(self, n_obs_steps: int):
        if n_obs_steps <= 0:
            raise ValueError(f"n_obs_steps must be positive, got {n_obs_steps}")
        self._buffers: dict[str, Deque[np.ndarray]] = {}
        self.n_obs_steps = int(n_obs_steps)

    def update(self, frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        stacked: dict[str, np.ndarray] = {}
        for key, value in frame.items():
            arr = np.asarray(value)
            if key not in self._buffers:
                self._buffers[key] = deque(maxlen=self.n_obs_steps)
            buf = self._buffers[key]
            if len(buf) == 0:
                while len(buf) < self.n_obs_steps:
                    buf.append(arr.copy())
            else:
                buf.append(arr.copy())
                while len(buf) < self.n_obs_steps:
                    buf.appendleft(buf[0].copy())
            stacked[key] = np.stack(list(buf), axis=0)
        return stacked

    def reset(self) -> None:
        self._buffers.clear()


def canonical_policy_obs(obs: dict[str, Any]) -> dict[str, Any]:
    assert_canonical_observation(obs)
    return {
        FRONT_IMAGE_KEY: np.asarray(obs[FRONT_IMAGE_KEY]),
        WRIST_IMAGE_KEY: np.asarray(obs[WRIST_IMAGE_KEY]),
        STATE_KEY: state7(obs),
    }


def _rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_quat().astype(np.float32)
