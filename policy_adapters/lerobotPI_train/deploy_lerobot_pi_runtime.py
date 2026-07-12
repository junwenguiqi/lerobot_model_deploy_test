from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from ..utils.common import ActionTarget, AdapterImageConfig, STATE_KEY
    from .lerobot_pi_adapter import DEFAULT_PI_TASK, LeRobotPIAdapter, LeRobotPIAdapterConfig
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from policy_adapters.utils.common import ActionTarget, AdapterImageConfig, STATE_KEY  # type: ignore
    from policy_adapters.lerobotPI_train.lerobot_pi_adapter import (  # type: ignore
        DEFAULT_PI_TASK,
        LeRobotPIAdapter,
        LeRobotPIAdapterConfig,
    )


@dataclass(frozen=True)
class PIRuntimeOutput:
    policy_input: dict[str, Any]
    action_raw: np.ndarray
    target: ActionTarget


class LeRobotPIRuntime:
    """Deployment helper for PI0 / PI0.5 checkpoints trained by train_lerobot_pi_minimal.py."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        preprocessor: Any,
        postprocessor: Any,
        adapter: LeRobotPIAdapter,
        device: str,
    ):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.adapter = adapter
        self.device = str(device)
        self.policy.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        lerobot_src: str | Path | None = None,
        device: str = "cuda",
        num_inference_steps: int | None = None,
        paligemma_tokenizer_path: str | Path | None = None,
    ) -> LeRobotPIRuntime:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if lerobot_src is not None:
            src = Path(lerobot_src).expanduser().resolve()
            if not src.exists():
                raise FileNotFoundError(src)
            sys.path.insert(0, str(src))

        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        checkpoint = _load_checkpoint(checkpoint_path, device)
        run_config = dict(checkpoint.get("run_config") or {})
        pi_config_data = dict(checkpoint.get("pi_config") or {})
        dataset_stats = dict(checkpoint.get("dataset_stats") or {})
        if not dataset_stats:
            stats_path = run_config.get("dataset_stats_path")
            if stats_path:
                dataset_stats = json.loads(Path(stats_path).expanduser().read_text(encoding="utf-8"))

        policy_type = str(run_config.get("policy_type", pi_config_data.get("type", "pi05")))
        image_shape = run_config.get("image_shape_chw") or [3, 224, 224]
        image_size = int(image_shape[-1])
        chunk_size = int(run_config.get("chunk_size", pi_config_data.get("chunk_size", 50)))
        n_action_steps = int(run_config.get("n_action_steps", pi_config_data.get("n_action_steps", chunk_size)))
        max_state_dim = int(run_config.get("max_state_dim", pi_config_data.get("max_state_dim", 32)))
        max_action_dim = int(run_config.get("max_action_dim", pi_config_data.get("max_action_dim", 32)))
        task = str(run_config.get("task_override") or DEFAULT_PI_TASK)

        adapter = LeRobotPIAdapter(
            LeRobotPIAdapterConfig(
                image=AdapterImageConfig(height=image_size, width=image_size),
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                task=task,
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
            "device": device,
            "dtype": str(run_config.get("dtype", pi_config_data.get("dtype", "float32"))),
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "max_state_dim": max_state_dim,
            "max_action_dim": max_action_dim,
            "image_resolution": (image_size, image_size),
            "num_inference_steps": int(
                num_inference_steps
                if num_inference_steps is not None
                else run_config.get("num_inference_steps", pi_config_data.get("num_inference_steps", 10))
            ),
            "paligemma_variant": str(
                run_config.get("paligemma_variant", pi_config_data.get("paligemma_variant", "gemma_2b"))
            ),
            "action_expert_variant": str(
                run_config.get("action_expert_variant", pi_config_data.get("action_expert_variant", "gemma_300m"))
            ),
            "freeze_vision_encoder": bool(
                run_config.get("freeze_vision_encoder", pi_config_data.get("freeze_vision_encoder", False))
            ),
            "train_expert_only": bool(
                run_config.get("train_expert_only", pi_config_data.get("train_expert_only", False))
            ),
            "gradient_checkpointing": bool(
                run_config.get("gradient_checkpointing", pi_config_data.get("gradient_checkpointing", False))
            ),
            "compile_model": False,
        }
        cfg_cls = PI05Config if policy_type in ("pi05", "pi0.5") else PI0Config
        policy_cls = PI05Policy if policy_type in ("pi05", "pi0.5") else PI0Policy
        cfg = cfg_cls(**cfg_kwargs)

        policy = policy_cls(cfg)
        policy.to(device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.reset()
        policy.eval()
        tokenizer_path = paligemma_tokenizer_path or run_config.get("paligemma_tokenizer_path")
        _patch_paligemma_tokenizer(tokenizer_path)
        preprocessor, postprocessor = make_pre_post_processors(policy_cfg=cfg, dataset_stats=dataset_stats)
        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            adapter=adapter,
            device=device,
        )

    @torch.no_grad()
    def predict(self, canonical_obs: dict[str, Any], *, task: str | None = None) -> PIRuntimeOutput:
        raw_state7 = np.asarray(canonical_obs[STATE_KEY], dtype=np.float32).reshape(7)
        policy_input = self.adapter.to_policy_input(canonical_obs, task=task, batched=False)
        policy_input = _tensorize_online_input(policy_input)
        processed = self.preprocessor(policy_input)

        action_normalized = self.policy.select_action(processed)
        action_raw = self.postprocessor(action_normalized)
        action_raw_np = _to_numpy(action_raw).reshape(-1, 7)[0].astype(np.float32)
        target = self.adapter.action_to_target(current_state7=raw_state7, action7=action_raw_np)
        return PIRuntimeOutput(policy_input=processed, action_raw=action_raw_np, target=target)

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()


def _tensorize_online_input(policy_input: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in policy_input.items():
        if key == "task":
            out[key] = value
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(dtype=torch.float32)
        else:
            out[key] = torch.as_tensor(value, dtype=torch.float32)
    return out


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


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_checkpoint(checkpoint_path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _patch_paligemma_tokenizer(local_path: str | Path | None) -> None:
    if not local_path:
        return
    path = Path(local_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"paligemma_tokenizer_path not found: {path}")

    from transformers import AutoTokenizer

    original_from_pretrained = AutoTokenizer.from_pretrained

    def redirected_from_pretrained(pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(pretrained_model_name_or_path) == "google/paligemma-3b-pt-224":
            pretrained_model_name_or_path = str(path)
            kwargs.setdefault("local_files_only", True)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    AutoTokenizer.from_pretrained = staticmethod(redirected_from_pretrained)  # type: ignore[method-assign]

