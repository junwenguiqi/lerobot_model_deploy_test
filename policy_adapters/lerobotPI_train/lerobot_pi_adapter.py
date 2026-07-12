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
        PolicyAdapterSpec,
        canonical_policy_obs,
        chw_float32_image,
        compose_relative_action,
    )


DEFAULT_PI_TASK = "fr3 pick and place with zed cameras"


@dataclass(frozen=True)
class LeRobotPIAdapterConfig:
    image: AdapterImageConfig = AdapterImageConfig(height=224, width=224)
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    task: str = DEFAULT_PI_TASK
    use_tactile: bool = False


class LeRobotPIAdapter:
    """Adapter for LeRobot PI0 / PI0.5 using this project's relative 7D action contract."""

    def __init__(self, cfg: LeRobotPIAdapterConfig | None = None):
        self.cfg = cfg or LeRobotPIAdapterConfig()
        if self.cfg.use_tactile:
            raise ValueError("vanilla LeRobot PI adapter does not consume tactile fields")
        if self.cfg.n_action_steps > self.cfg.chunk_size:
            raise ValueError("n_action_steps cannot be greater than chunk_size")

    def spec(self) -> PolicyAdapterSpec:
        return PolicyAdapterSpec(
            name="lerobot_pi0_pi05",
            input_keys=(FRONT_IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY),
            output_key=ACTION_KEY,
            state_dim=7,
            action_dim=7,
            image_shape_chw=self.cfg.image.chw_shape,
            action_semantics="step-wise relative end-effector delta",
            tactile_used=False,
            notes=(
                "PI0 and PI0.5 use one current observation plus a language task string.",
                "action_delta_indices is range(chunk_size).",
                "PI0 uses mean/std state-action normalization; PI0.5 uses q01/q99 quantile normalization.",
                "Tactile columns may remain in the dataset but are not included in input_features.",
            ),
        )

    def lerobot_policy_features(self) -> dict[str, dict[str, Any]]:
        return {
            FRONT_IMAGE_KEY: {"type": "VISUAL", "shape": self.cfg.image.chw_shape},
            WRIST_IMAGE_KEY: {"type": "VISUAL", "shape": self.cfg.image.chw_shape},
            STATE_KEY: {"type": "STATE", "shape": (7,)},
            ACTION_KEY: {"type": "ACTION", "shape": (7,)},
        }

    def lerobot_input_features(self) -> dict[str, dict[str, Any]]:
        features = self.lerobot_policy_features()
        return {key: features[key] for key in (FRONT_IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY)}

    def lerobot_output_features(self) -> dict[str, dict[str, Any]]:
        return {ACTION_KEY: self.lerobot_policy_features()[ACTION_KEY]}

    def delta_timestamps(self, fps: float) -> dict[str, list[float]]:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        return {ACTION_KEY: [i / float(fps) for i in range(self.cfg.chunk_size)]}

    def to_policy_input(
        self,
        canonical_obs: dict[str, Any],
        *,
        task: str | None = None,
        batched: bool = False,
    ) -> dict[str, Any]:
        """Convert online canonical observation into a PI preprocessor input.

        Leave ``batched=False`` when passing the result to LeRobot's PI preprocessor, because the
        processor already adds a batch dimension for online single-frame observations.
        """
        obs = canonical_policy_obs(canonical_obs)
        frame: dict[str, Any] = {
            FRONT_IMAGE_KEY: chw_float32_image(obs[FRONT_IMAGE_KEY], self.cfg.image),
            WRIST_IMAGE_KEY: chw_float32_image(obs[WRIST_IMAGE_KEY], self.cfg.image),
            STATE_KEY: np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(7),
            "task": str(task or self.cfg.task),
        }
        if not batched:
            return frame
        return {
            key: ([value] if key == "task" else value[None, ...])
            for key, value in frame.items()
        }

    def to_training_batch(
        self,
        batch: dict[str, Any],
        *,
        device: str | None = None,
        task_override: str | None = None,
    ) -> dict[str, Any]:
        """Convert a LeRobotDataset/DataLoader batch into the raw batch expected by PI processors."""
        import torch
        import torch.nn.functional as F

        front = _as_batched_chw_float(batch[FRONT_IMAGE_KEY], FRONT_IMAGE_KEY, torch=torch)
        wrist = _as_batched_chw_float(batch[WRIST_IMAGE_KEY], WRIST_IMAGE_KEY, torch=torch)
        state = _as_float_tensor(batch[STATE_KEY], STATE_KEY, torch=torch)
        action = _as_float_tensor(batch[ACTION_KEY], ACTION_KEY, torch=torch)

        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[-1] != 7:
            raise ValueError(f"{STATE_KEY} must have shape [B, 7], got {tuple(state.shape)}")

        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3 or action.shape[-2:] != (self.cfg.chunk_size, 7):
            raise ValueError(
                f"{ACTION_KEY} must have shape [B, {self.cfg.chunk_size}, 7], got {tuple(action.shape)}. "
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

        front = _resize_batched_chw(front, self.cfg.image.hwc_shape[:2], F=F)
        wrist = _resize_batched_chw(wrist, self.cfg.image.hwc_shape[:2], F=F)

        out: dict[str, Any] = {
            FRONT_IMAGE_KEY: front.contiguous(),
            WRIST_IMAGE_KEY: wrist.contiguous(),
            STATE_KEY: state.contiguous(),
            ACTION_KEY: action.contiguous(),
            f"{ACTION_KEY}_is_pad": action_is_pad.contiguous(),
            "task": _task_batch(batch.get("task"), batch_size=int(action.shape[0]), fallback=task_override or self.cfg.task),
        }
        if task_override is not None:
            out["task"] = [str(task_override)] * int(action.shape[0])
        if device is not None:
            for key, value in list(out.items()):
                if hasattr(value, "to"):
                    out[key] = value.to(device)
        return out

    def action_to_target(self, current_state7: np.ndarray, action7: np.ndarray) -> ActionTarget:
        return compose_relative_action(current_state7=current_state7, action7=action7)


def _as_float_tensor(value: Any, name: str, *, torch: Any) -> Any:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    else:
        tensor = tensor.to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _as_batched_chw_float(value: Any, name: str, *, torch: Any) -> Any:
    tensor = _as_float_tensor(value, name, torch=torch)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 5 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W], got {tuple(tensor.shape)}")
    if tensor.shape[1] != 3:
        raise ValueError(f"{name} must have 3 channels in CHW layout, got {tuple(tensor.shape)}")
    if float(tensor.max().detach().cpu()) > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _resize_batched_chw(value: Any, hw_shape: tuple[int, int], *, F: Any) -> Any:
    height, width = int(hw_shape[0]), int(hw_shape[1])
    if value.shape[-2:] == (height, width):
        return value
    return F.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)


def _task_batch(value: Any, *, batch_size: int, fallback: str) -> list[str]:
    if value is None:
        return [str(fallback)] * batch_size

    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
    except ModuleNotFoundError:
        pass

    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return [str(item) for item in value]
        if len(value) == 1:
            return [str(value[0])] * batch_size
    return [str(value)] * batch_size

