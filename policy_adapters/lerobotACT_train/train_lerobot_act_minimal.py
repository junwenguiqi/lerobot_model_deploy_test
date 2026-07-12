from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import typing
from pathlib import Path
from typing import Any

import torch
from typing_extensions import Unpack as TypingExtensionsUnpack

if not hasattr(typing, "Unpack"):
    typing.Unpack = TypingExtensionsUnpack  # type: ignore[attr-defined]

try:
    from ..utils.common import AdapterImageConfig
    from .lerobot_act_adapter import LeRobotACTAdapter, LeRobotACTAdapterConfig
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from policy_adapters.utils.common import AdapterImageConfig  # type: ignore
    from policy_adapters.lerobotACT_train.lerobot_act_adapter import (  # type: ignore
        LeRobotACTAdapter,
        LeRobotACTAdapterConfig,
    )


DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parents[2]
    / "dataset_tactile"
    / "fr3_zed_lerobot_tashan_v3_train56"
)
DEFAULT_OUTPUT_DIR = Path.cwd() / "outputs" / "lerobot_act_minimal"


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Run this script from a Python 3.10+ environment; Python 3.12+ is preferred.")

    parser = argparse.ArgumentParser(
        description="Minimal LeRobot ACT training loop using project policy adapter semantics."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Local LeRobot dataset root")
    parser.add_argument("--repo-id", default="local/fr3_zed_lerobot_tashan_v3_train56")
    parser.add_argument("--lerobot-src", default=_guess_lerobot_src(), help="Path to ./lerobot/src")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--episode", type=int, action="append", default=None, help="Episode index to load")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--dim-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--n-encoder-layers", type=int, default=1)
    parser.add_argument("--n-vae-encoder-layers", type=int, default=1)
    parser.add_argument("--no-vae", action="store_true")
    parser.add_argument(
        "--pretrained-backbone",
        default="none",
        choices=("none", "imagenet"),
        help="Use none to avoid downloading torchvision weights during minimal tests.",
    )
    parser.add_argument("--save-every", type=int, default=0, help="Optional checkpoint cadence in steps")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="lerobot-act", help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity (team/user)")
    parser.add_argument("--wandb-name", default=None, help="W&B run name")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode")
    parser.add_argument(
        "--normalization",
        default="mean_std",
        choices=("mean_std", "none"),
        help="Normalize state/action with dataset stats before ACTPolicy.forward.",
    )
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")

    _set_local_hf_cache()
    _add_lerobot_to_path(args.lerobot_src)
    _install_lerobot_namespace_shims(Path(args.lerobot_src).expanduser().resolve())

    try:
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency while importing local LeRobot: {exc}. "
            "Use a Python 3.12 LeRobot environment for formal training."
        ) from exc

    _set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    dataset_stats = _read_stats(dataset_root) if args.normalization == "mean_std" else {}

    adapter = LeRobotACTAdapter(
        LeRobotACTAdapterConfig(
            image=AdapterImageConfig(height=args.image_size, width=args.image_size),
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            use_tactile=False,
        )
    )

    fps = float(_read_info(dataset_root)["fps"])
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        episodes=args.episode,
        delta_timestamps=adapter.delta_timestamps(fps),
        video_backend=args.video_backend,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=str(args.device).startswith("cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    if len(loader) == 0:
        raise RuntimeError("DataLoader is empty; reduce --batch-size or check selected episodes.")

    feature_type = {
        "VISUAL": FeatureType.VISUAL,
        "STATE": FeatureType.STATE,
        "ACTION": FeatureType.ACTION,
    }
    input_features = _to_policy_features(adapter.lerobot_input_features(), PolicyFeature, feature_type)
    output_features = _to_policy_features(adapter.lerobot_output_features(), PolicyFeature, feature_type)

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        device=args.device,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        n_encoder_layers=args.n_encoder_layers,
        n_vae_encoder_layers=args.n_vae_encoder_layers,
        use_vae=not args.no_vae,
        pretrained_backbone_weights=(
            "ResNet18_Weights.IMAGENET1K_V1" if args.pretrained_backbone == "imagenet" else None
        ),
        optimizer_lr=args.lr,
        optimizer_weight_decay=args.weight_decay,
        optimizer_lr_backbone=args.lr,
    )
    policy = ACTPolicy(cfg).to(args.device)
    policy.train()

    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr, weight_decay=args.weight_decay)

    stats_path = output_dir / "dataset_stats.json"
    run_config = {
        "dataset_root": str(dataset_root),
        "dataset_frames": int(dataset.num_frames),
        "dataset_episodes": int(dataset.num_episodes),
        "fps": fps,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "chunk_size": int(args.chunk_size),
        "n_action_steps": int(args.n_action_steps),
        "image_shape_chw": list(adapter.spec().image_shape_chw),
        "input_feature_keys": list(cfg.input_features.keys()),
        "output_feature_keys": list(cfg.output_features.keys()),
        "device": str(args.device),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "use_vae": bool(cfg.use_vae),
        "normalization": args.normalization,
        "dataset_stats_path": str(stats_path) if dataset_stats else None,
    }
    _write_json(output_dir / "run_config.json", run_config)
    if dataset_stats:
        _write_json(stats_path, dataset_stats)

    use_wandb = args.wandb
    if use_wandb:
        import wandb

        wandb_kwargs: dict[str, Any] = dict(
            project=args.wandb_project,
            config=run_config,
            dir=str(output_dir),
        )
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_name:
            wandb_kwargs["name"] = args.wandb_name
        if args.wandb_offline:
            wandb_kwargs["mode"] = "offline"
        wandb.init(**wandb_kwargs)
        wandb.watch(policy, log="gradients", log_freq=max(1, args.steps // 20))

    log_path = output_dir / "train_log.jsonl"
    first_shapes: dict[str, list[int]] | None = None
    step = 0
    data_iter = iter(loader)
    start = time.perf_counter()
    last_loss = math.nan

    with log_path.open("w", encoding="utf-8") as log_f:
        while step < args.steps:
            try:
                raw_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                raw_batch = next(data_iter)

            step += 1
            step_start = time.perf_counter()
            batch = adapter.to_training_batch(raw_batch, device=args.device)
            if args.normalization == "mean_std":
                batch = adapter.normalize_training_batch(batch, dataset_stats)
            if first_shapes is None:
                first_shapes = {key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")}

            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy.forward(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
            optimizer.step()

            last_loss = float(loss.detach().cpu())
            entry = {
                "step": step,
                "loss": last_loss,
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "step_s": time.perf_counter() - step_start,
                "loss_dict": {key: float(value) for key, value in loss_dict.items()},
            }
            log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log_f.flush()
            print(json.dumps(entry, ensure_ascii=False))
            if use_wandb:
                wandb.log(entry, step=step)

            if args.save_every > 0 and step % args.save_every == 0:
                _save_checkpoint(
                    output_dir / f"checkpoint_step_{step:06d}.pt",
                    policy,
                    optimizer,
                    cfg,
                    step,
                    run_config,
                    dataset_stats,
                )

    final_ckpt = output_dir / "checkpoint_last.pt"
    _save_checkpoint(final_ckpt, policy, optimizer, cfg, step, run_config, dataset_stats)
    if use_wandb:
        wandb.finish()
    summary = {
        "ok": True,
        "output_dir": str(output_dir),
        "checkpoint": str(final_ckpt),
        "steps": int(step),
        "last_loss": last_loss,
        "elapsed_s": time.perf_counter() - start,
        "first_batch_shapes": first_shapes,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _read_info(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "meta" / "info.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_stats(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "meta" / "stats.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _guess_lerobot_src() -> str:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "lerobot" / "src"
        if candidate.exists():
            return str(candidate)
    candidate = Path.cwd() / "lerobot" / "src"
    return str(candidate)


def _add_lerobot_to_path(lerobot_src: str | None) -> None:
    if not lerobot_src:
        return
    path = Path(lerobot_src).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LeRobot src path not found: {path}")
    sys.path.insert(0, str(path))


def _set_local_hf_cache() -> None:
    cache_root = Path.cwd() / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    for key in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def _install_lerobot_namespace_shims(lerobot_src: Path) -> None:
    """Bypass Python 3.12-only eager imports when running quick checks from older interpreters."""
    if sys.version_info >= (3, 12):
        return

    import types

    pretrained_mod = types.ModuleType("lerobot.policies.pretrained")

    class PreTrainedPolicy(torch.nn.Module):
        config_class = None
        name = None

        def __init__(self, config: Any, *inputs: Any, **kwargs: Any):
            super().__init__()
            self.config = config

    pretrained_mod.PreTrainedPolicy = PreTrainedPolicy
    sys.modules.setdefault("lerobot.policies.pretrained", pretrained_mod)

    optim_pkg = types.ModuleType("lerobot.optim")
    optim_pkg.__path__ = [str(lerobot_src / "lerobot" / "optim")]  # type: ignore[attr-defined]
    sys.modules.setdefault("lerobot.optim", optim_pkg)

    optimizers_mod = types.ModuleType("lerobot.optim.optimizers")

    class OptimizerConfig:
        pass

    class AdamWConfig(OptimizerConfig):
        def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-2, grad_clip_norm: float = 10.0):
            self.lr = lr
            self.weight_decay = weight_decay
            self.grad_clip_norm = grad_clip_norm

    optimizers_mod.OptimizerConfig = OptimizerConfig
    optimizers_mod.AdamWConfig = AdamWConfig
    sys.modules.setdefault("lerobot.optim.optimizers", optimizers_mod)

    schedulers_mod = types.ModuleType("lerobot.optim.schedulers")

    class LRSchedulerConfig:
        pass

    schedulers_mod.LRSchedulerConfig = LRSchedulerConfig
    sys.modules.setdefault("lerobot.optim.schedulers", schedulers_mod)

    shims = {
        "lerobot.datasets": lerobot_src / "lerobot" / "datasets",
        "lerobot.policies": lerobot_src / "lerobot" / "policies",
        "lerobot.policies.act": lerobot_src / "lerobot" / "policies" / "act",
    }
    for name, path in shims.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        module.__package__ = name
        sys.modules[name] = module


def _save_checkpoint(
    path: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    step: int,
    run_config: dict[str, Any],
    dataset_stats: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "act_config": _jsonable_config(cfg),
            "run_config": run_config,
            "dataset_stats": dataset_stats,
        },
        path,
    )


def _jsonable_config(cfg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(cfg).items():
        if key.startswith("_"):
            continue
        out[key] = _jsonable(value)
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "type") and hasattr(value, "shape"):
        return {"type": str(value.type), "shape": list(value.shape)}
    return str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
