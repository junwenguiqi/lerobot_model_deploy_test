from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from ..utils.common import (
        ACTION_KEY,
        FRONT_IMAGE_KEY,
        STATE_KEY,
        WRIST_IMAGE_KEY,
        AdapterImageConfig,
        ActionTarget,
        ImageRoiConfig,
        PolicyAdapterSpec,
        canonical_policy_obs,
        chw_float32_image,
        compose_relative_action,
    )
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from policy_adapters.utils.common import (  # type: ignore
        ACTION_KEY,
        FRONT_IMAGE_KEY,
        STATE_KEY,
        WRIST_IMAGE_KEY,
        AdapterImageConfig,
        ActionTarget,
        ImageRoiConfig,
        PolicyAdapterSpec,
        canonical_policy_obs,
        chw_float32_image,
        compose_relative_action,
    )


@dataclass(frozen=True)
class LeRobotDPAdapterConfig:
    image: AdapterImageConfig = AdapterImageConfig()
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    use_tactile: bool = False
    front_roi: ImageRoiConfig | None = None


class LeRobotDPAdapter:
    """Adapter for LeRobot DiffusionPolicy with step-wise relative EE actions."""

    def __init__(self, cfg: LeRobotDPAdapterConfig | None = None):
        self.cfg = cfg or LeRobotDPAdapterConfig()
        if self.cfg.use_tactile:
            raise ValueError("vanilla LeRobot DP adapter does not consume tactile fields")

    def spec(self) -> PolicyAdapterSpec:
        return PolicyAdapterSpec(
            name="lerobot_diffusion",
            input_keys=(FRONT_IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY),
            output_key=ACTION_KEY,
            state_dim=7,
            action_dim=7,
            image_shape_chw=self.cfg.image.chw_shape,
            action_semantics="step-wise relative end-effector delta",
            tactile_used=False,
            notes=(
                "LeRobot DiffusionPolicy caches n_obs_steps observations internally at inference.",
                "All image feature shapes must match.",
                "Optional fixed front ROI is applied before adapter resize when configured.",
                "action_delta_indices starts at 1 - n_obs_steps and spans horizon.",
            ),
        )

    def lerobot_policy_features(self) -> dict[str, dict[str, Any]]:
        return {
            FRONT_IMAGE_KEY: {"type": "VISUAL", "shape": self.cfg.image.chw_shape},
            WRIST_IMAGE_KEY: {"type": "VISUAL", "shape": self.cfg.image.chw_shape},
            STATE_KEY: {"type": "STATE", "shape": (7,)},
            ACTION_KEY: {"type": "ACTION", "shape": (7,)},
        }

    def delta_timestamps(self, fps: float) -> dict[str, list[float]]:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        obs_indices = list(range(1 - self.cfg.n_obs_steps, 1))
        action_indices = list(range(1 - self.cfg.n_obs_steps, 1 - self.cfg.n_obs_steps + self.cfg.horizon))
        return {
            FRONT_IMAGE_KEY: [i / float(fps) for i in obs_indices],
            WRIST_IMAGE_KEY: [i / float(fps) for i in obs_indices],
            STATE_KEY: [i / float(fps) for i in obs_indices],
            ACTION_KEY: [i / float(fps) for i in action_indices],
        }

    def to_policy_input(self, canonical_obs: dict[str, Any], *, batched: bool = True) -> dict[str, np.ndarray]:
        obs = canonical_policy_obs(canonical_obs)
        frame = {
            FRONT_IMAGE_KEY: chw_float32_image(
                obs[FRONT_IMAGE_KEY],
                self.cfg.image,
                roi=self.cfg.front_roi,
                name=FRONT_IMAGE_KEY,
            ),
            WRIST_IMAGE_KEY: chw_float32_image(obs[WRIST_IMAGE_KEY], self.cfg.image),
            STATE_KEY: np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(7),
        }
        if not batched:
            return frame
        return {key: value[None, ...] for key, value in frame.items()}

    def to_training_batch(self, batch: dict[str, Any], *, device: str | None = None) -> dict[str, Any]:
        """Convert a LeRobotDataset/DataLoader batch into tensors DiffusionPolicy.forward expects."""
        import torch
        import torch.nn.functional as F

        front = _as_batched_time_chw_float(
            batch[FRONT_IMAGE_KEY], FRONT_IMAGE_KEY, n_obs_steps=self.cfg.n_obs_steps, torch=torch
        )
        wrist = _as_batched_time_chw_float(
            batch[WRIST_IMAGE_KEY], WRIST_IMAGE_KEY, n_obs_steps=self.cfg.n_obs_steps, torch=torch
        )
        state = _as_float_tensor(batch[STATE_KEY], STATE_KEY, torch=torch)
        action = _as_float_tensor(batch[ACTION_KEY], ACTION_KEY, torch=torch)

        if state.ndim == 1:
            state = state.reshape(1, 1, 7)
        elif state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3 or state.shape[-2:] != (self.cfg.n_obs_steps, 7):
            raise ValueError(
                f"{STATE_KEY} must have shape [B, {self.cfg.n_obs_steps}, 7], got {tuple(state.shape)}. "
                "Pass delta_timestamps=adapter.delta_timestamps(fps) when constructing LeRobotDataset."
            )

        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3 or action.shape[-2:] != (self.cfg.horizon, 7):
            raise ValueError(
                f"{ACTION_KEY} must have shape [B, {self.cfg.horizon}, 7], got {tuple(action.shape)}. "
                "Pass delta_timestamps=adapter.delta_timestamps(fps) when constructing LeRobotDataset."
            )

        action_is_pad = batch.get(f"{ACTION_KEY}_is_pad")
        if action_is_pad is None:
            action_is_pad = torch.zeros(action.shape[:2], dtype=torch.bool)
        else:
            action_is_pad = torch.as_tensor(action_is_pad, dtype=torch.bool)
            if action_is_pad.ndim == 1:
                action_is_pad = action_is_pad.unsqueeze(0)
        if action_is_pad.shape != action.shape[:2]:
            raise ValueError(
                f"{ACTION_KEY}_is_pad must have shape {tuple(action.shape[:2])}, "
                f"got {tuple(action_is_pad.shape)}"
            )

        front = _crop_batched_time_chw(front, self.cfg.front_roi)
        front = _resize_batched_time_chw(front, self.cfg.image.hwc_shape[:2], F=F)
        wrist = _resize_batched_time_chw(wrist, self.cfg.image.hwc_shape[:2], F=F)

        out = {
            FRONT_IMAGE_KEY: front.contiguous(),
            WRIST_IMAGE_KEY: wrist.contiguous(),
            STATE_KEY: state.contiguous(),
            ACTION_KEY: action.contiguous(),
            f"{ACTION_KEY}_is_pad": action_is_pad.contiguous(),
        }
        if device is not None:
            out = {key: value.to(device) for key, value in out.items()}
        return out

    def normalize_training_batch(self, batch: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
        """Apply LeRobot DiffusionPolicy default min/max normalization to state and action."""
        out = dict(batch)
        out[STATE_KEY] = normalize_feature_min_max(out[STATE_KEY], stats, STATE_KEY)
        out[ACTION_KEY] = normalize_feature_min_max(out[ACTION_KEY], stats, ACTION_KEY)
        return out

    def normalize_policy_input(self, policy_input: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
        """Normalize online current observation with the same stats used during training."""
        out = dict(policy_input)
        out[STATE_KEY] = normalize_feature_min_max(out[STATE_KEY], stats, STATE_KEY)
        return out

    def unnormalize_action(self, action: Any, stats: dict[str, Any]) -> Any:
        """Convert normalized DiffusionPolicy output back to raw relative action units."""
        return unnormalize_feature_min_max(action, stats, ACTION_KEY)

    def action_to_target(self, current_state7: np.ndarray, action7: np.ndarray) -> ActionTarget:
        return compose_relative_action(current_state7=current_state7, action7=action7)


def normalize_feature_min_max(value: Any, stats: dict[str, Any], key: str, eps: float = 1e-8) -> Any:
    return _apply_min_max(value, stats, key, inverse=False, eps=eps)


def unnormalize_feature_min_max(value: Any, stats: dict[str, Any], key: str, eps: float = 1e-8) -> Any:
    return _apply_min_max(value, stats, key, inverse=True, eps=eps)


def _apply_min_max(value: Any, stats: dict[str, Any], key: str, *, inverse: bool, eps: float) -> Any:
    if key not in stats:
        raise KeyError(f"missing normalization stats for {key!r}")
    entry = stats[key]
    if "min" not in entry or "max" not in entry:
        raise KeyError(f"normalization stats for {key!r} must include min and max")

    try:
        import torch

        if isinstance(value, torch.Tensor):
            min_val = torch.as_tensor(entry["min"], dtype=value.dtype, device=value.device)
            max_val = torch.as_tensor(entry["max"], dtype=value.dtype, device=value.device)
            while min_val.ndim < value.ndim:
                min_val = min_val.unsqueeze(0)
                max_val = max_val.unsqueeze(0)
            denom = torch.where(
                max_val == min_val,
                torch.tensor(eps, dtype=value.dtype, device=value.device),
                max_val - min_val,
            )
            if inverse:
                return (value + 1.0) / 2.0 * denom + min_val
            return 2.0 * (value - min_val) / denom - 1.0
    except ModuleNotFoundError:
        pass

    arr = np.asarray(value, dtype=np.float32)
    min_val = np.asarray(entry["min"], dtype=np.float32)
    max_val = np.asarray(entry["max"], dtype=np.float32)
    while min_val.ndim < arr.ndim:
        min_val = np.expand_dims(min_val, axis=0)
        max_val = np.expand_dims(max_val, axis=0)
    denom = np.where(max_val == min_val, np.float32(eps), max_val - min_val)
    if inverse:
        return (arr + np.float32(1.0)) / np.float32(2.0) * denom + min_val
    return np.float32(2.0) * (arr - min_val) / denom - np.float32(1.0)


def _as_float_tensor(value: Any, name: str, *, torch: Any) -> Any:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    else:
        tensor = tensor.to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _as_batched_time_chw_float(value: Any, name: str, *, n_obs_steps: int, torch: Any) -> Any:
    tensor = _as_float_tensor(value, name, torch=torch)
    if tensor.ndim == 3:
        tensor = tensor.reshape(1, 1, *tensor.shape)
    elif tensor.ndim == 4:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 5:
        raise ValueError(f"{name} must have shape [B, T, C, H, W], got {tuple(tensor.shape)}")
    if tensor.shape[1] != int(n_obs_steps):
        raise ValueError(f"{name} must have n_obs_steps={n_obs_steps} at dim 1, got {tuple(tensor.shape)}")
    if tensor.shape[2] != 3:
        raise ValueError(f"{name} must have 3 channels in CHW layout, got {tuple(tensor.shape)}")
    if float(tensor.max().detach().cpu()) > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _resize_batched_time_chw(value: Any, hw_shape: tuple[int, int], *, F: Any) -> Any:
    height, width = int(hw_shape[0]), int(hw_shape[1])
    if value.shape[-2:] == (height, width):
        return value
    batch, time = value.shape[:2]
    flat = value.reshape(batch * time, *value.shape[2:])
    resized = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
    return resized.reshape(batch, time, *resized.shape[1:])


def _crop_batched_time_chw(value: Any, roi: ImageRoiConfig | None) -> Any:
    if roi is None:
        return value
    left, top, right, bottom = roi.box
    height, width = int(value.shape[-2]), int(value.shape[-1])
    roi.validate((width, height), name="front_roi")
    return value[..., top:bottom, left:right]

