from __future__ import annotations

import json
import sys
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from typing_extensions import Unpack as TypingExtensionsUnpack

if not hasattr(typing, "Unpack"):
    typing.Unpack = TypingExtensionsUnpack  # type: ignore[attr-defined]

try:
    from ..utils.common import ActionTarget, AdapterImageConfig, STATE_KEY
    from .lerobot_act_adapter import LeRobotACTAdapter, LeRobotACTAdapterConfig
    from .train_lerobot_act_minimal import _install_lerobot_namespace_shims
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from policy_adapters.utils.common import ActionTarget, AdapterImageConfig, STATE_KEY  # type: ignore
    from policy_adapters.lerobotACT_train.lerobot_act_adapter import (  # type: ignore
        LeRobotACTAdapter,
        LeRobotACTAdapterConfig,
    )
    from policy_adapters.lerobotACT_train.train_lerobot_act_minimal import (  # type: ignore
        _install_lerobot_namespace_shims,
    )


@dataclass(frozen=True)
class ACTRuntimeOutput:
    policy_input: dict[str, Any]
    action_normalized: np.ndarray
    action_raw: np.ndarray
    target: ActionTarget


class LeRobotACTRuntime:
    """Deployment helper for ACT checkpoints trained by train_lerobot_act_minimal.py."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        adapter: LeRobotACTAdapter,
        dataset_stats: dict[str, Any],
        normalization: str,
        device: str,
    ):
        self.policy = policy
        self.adapter = adapter
        self.dataset_stats = dataset_stats
        self.normalization = str(normalization)
        self.device = str(device)
        self.policy.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        lerobot_src: str | Path | None = None,
        device: str = "cuda",
    ) -> LeRobotACTRuntime:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if lerobot_src is not None:
            src = Path(lerobot_src).expanduser().resolve()
            if not src.exists():
                raise FileNotFoundError(src)
            sys.path.insert(0, str(src))
            _install_lerobot_namespace_shims(src)

        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        checkpoint = _load_checkpoint(checkpoint_path, device)
        run_config = dict(checkpoint.get("run_config") or {})
        act_config_data = dict(checkpoint.get("act_config") or {})
        dataset_stats = dict(checkpoint.get("dataset_stats") or {})
        if not dataset_stats:
            stats_path = run_config.get("dataset_stats_path")
            if stats_path:
                dataset_stats = json.loads(Path(stats_path).expanduser().read_text(encoding="utf-8"))

        image_shape = run_config.get("image_shape_chw") or [3, 256, 256]
        image_size = int(image_shape[-1])
        chunk_size = int(run_config.get("chunk_size", act_config_data.get("chunk_size", 100)))
        n_action_steps = int(run_config.get("n_action_steps", act_config_data.get("n_action_steps", chunk_size)))
        normalization = str(run_config.get("normalization", "none"))

        adapter = LeRobotACTAdapter(
            LeRobotACTAdapterConfig(
                image=AdapterImageConfig(height=image_size, width=image_size),
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                use_tactile=False,
            )
        )

        feature_type = {
            "VISUAL": FeatureType.VISUAL,
            "STATE": FeatureType.STATE,
            "ACTION": FeatureType.ACTION,
        }
        input_features = _to_policy_features(adapter.lerobot_input_features(), PolicyFeature, feature_type)
        output_features = _to_policy_features(adapter.lerobot_output_features(), PolicyFeature, feature_type)

        cfg_kwargs = {
            "input_features": input_features,
            "output_features": output_features,
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "device": device,
        }
        for key in (
            "dim_model",
            "n_heads",
            "dim_feedforward",
            "n_encoder_layers",
            "n_vae_encoder_layers",
            "use_vae",
            "pretrained_backbone_weights",
        ):
            if key in act_config_data:
                cfg_kwargs[key] = act_config_data[key]
        cfg = ACTConfig(**cfg_kwargs)

        policy = ACTPolicy(cfg).to(device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        return cls(
            policy=policy,
            adapter=adapter,
            dataset_stats=dataset_stats,
            normalization=normalization,
            device=device,
        )

    @torch.no_grad()
    def predict(self, canonical_obs: dict[str, Any]) -> ACTRuntimeOutput:
        raw_state7 = np.asarray(canonical_obs[STATE_KEY], dtype=np.float32).reshape(7)
        policy_input = self.adapter.to_policy_input(canonical_obs, batched=True)
        policy_input = {key: _to_torch(value, self.device) for key, value in policy_input.items()}
        if self.normalization == "mean_std":
            policy_input = self.adapter.normalize_policy_input(policy_input, self.dataset_stats)

        action_norm = self.policy.select_action(policy_input)
        action_raw = action_norm
        if self.normalization == "mean_std":
            action_raw = self.adapter.unnormalize_action(action_norm, self.dataset_stats)

        action_norm_np = _to_numpy(action_norm).reshape(-1, 7)[0].astype(np.float32)
        action_raw_np = _to_numpy(action_raw).reshape(-1, 7)[0].astype(np.float32)
        target = self.adapter.action_to_target(current_state7=raw_state7, action7=action_raw_np)
        return ACTRuntimeOutput(
            policy_input=policy_input,
            action_normalized=action_norm_np,
            action_raw=action_raw_np,
            target=target,
        )

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()


def _to_policy_features(
    features: dict[str, dict[str, Any]],
    policy_feature_cls: Any,
    feature_type: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: policy_feature_cls(
            type=feature_type[str(spec["type"])],
            shape=tuple(int(x) for x in spec["shape"]),
        )
        for key, spec in features.items()
    }


def _to_torch(value: Any, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_checkpoint(checkpoint_path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)

