from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

try:
    from ..utils.common import ACTION_KEY, STATE_KEY, AdapterImageConfig
    from .lerobot_pi_adapter import DEFAULT_PI_TASK, LeRobotPIAdapter, LeRobotPIAdapterConfig
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from policy_adapters.utils.common import ACTION_KEY, STATE_KEY, AdapterImageConfig  # type: ignore
    from policy_adapters.lerobotPI_train.lerobot_pi_adapter import (  # type: ignore
        DEFAULT_PI_TASK,
        LeRobotPIAdapter,
        LeRobotPIAdapterConfig,
    )


DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parents[2]
    / "dataset_tactile"
    / "fr3_zed_lerobot_tashan_v3_train56"
)
DEFAULT_OUTPUT_DIR = Path.cwd() / "outputs" / "lerobot_pi_minimal"


def main() -> None:
    if sys.version_info < (3, 12):
        raise SystemExit("PI0/PI0.5 in this local LeRobot tree should be run from Python 3.12+.")

    parser = argparse.ArgumentParser(
        description="Minimal LeRobot PI0 / PI0.5 training loop using project adapter semantics."
    )
    parser.add_argument("--policy-type", default="pi05", choices=("pi0", "pi05"))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Local LeRobot dataset root")
    parser.add_argument("--repo-id", default="local/fr3_zed_lerobot_tashan_v3_train56")
    parser.add_argument("--lerobot-src", default=_guess_lerobot_src(), help="Path to ./lerobot/src")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-state-dim", type=int, default=32)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--episode", type=int, action="append", default=None, help="Episode index to load")
    parser.add_argument("--task", default=None, help="Override dataset task text for every sample")
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--optimizer-beta1", type=float, default=0.9)
    parser.add_argument("--optimizer-beta2", type=float, default=0.95)
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--scheduler-warmup-steps", type=int, default=1000)
    parser.add_argument("--scheduler-decay-steps", type=int, default=30000)
    parser.add_argument("--scheduler-decay-lr", type=float, default=2.5e-6)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--paligemma-variant", default="gemma_2b", choices=("gemma_300m", "gemma_2b"))
    parser.add_argument("--action-expert-variant", default="gemma_300m", choices=("gemma_300m", "gemma_2b"))
    parser.add_argument("--pretrained-path", default=None, help="Optional HF repo id or local PI checkpoint dir")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download pretrained files")
    parser.add_argument("--hf-home", default=None, help="Optional HF_HOME cache directory")
    parser.add_argument("--hf-endpoint", default=None, help="Optional Hugging Face endpoint/mirror URL")
    parser.add_argument("--hf-offline", action="store_true", help="Force HF/transformers offline cache mode")
    parser.add_argument(
        "--paligemma-tokenizer-path",
        default=None,
        help="Optional local directory for google/paligemma-3b-pt-224 tokenizer/config files",
    )
    parser.add_argument("--freeze-vision-encoder", action="store_true")
    parser.add_argument("--train-expert-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--save-every", type=int, default=0, help="Optional checkpoint cadence in steps")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="lerobot-pi", help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity (team/user)")
    parser.add_argument("--wandb-name", default=None, help="W&B run name")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    if args.n_action_steps > args.chunk_size:
        raise ValueError("--n-action-steps must be <= --chunk-size")

    _configure_hf_env(args)
    _log("HF cache/env configured")
    _add_lerobot_to_path(args.lerobot_src)
    _log(f"LeRobot src added: {args.lerobot_src}")

    try:
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency while importing local LeRobot PI policy: {exc}. "
            "Use the armpy312 Python 3.12 environment with LeRobot dependencies installed."
        ) from exc
    _log("LeRobot PI imports ok")

    _set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    _log(f"Dataset root: {dataset_root}")

    adapter = LeRobotPIAdapter(
        LeRobotPIAdapterConfig(
            image=AdapterImageConfig(height=args.image_size, width=args.image_size),
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            max_state_dim=args.max_state_dim,
            max_action_dim=args.max_action_dim,
            task=args.task or DEFAULT_PI_TASK,
            use_tactile=False,
        )
    )

    dataset_stats = _read_stats(dataset_root)
    if args.policy_type == "pi05":
        _require_quantile_stats(dataset_stats)
    else:
        _require_mean_std_stats(dataset_stats)
    _log("Dataset stats ok")

    fps = float(_read_info(dataset_root)["fps"])
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        episodes=args.episode,
        delta_timestamps=adapter.delta_timestamps(fps),
        video_backend=args.video_backend,
    )
    _log(f"Dataset loaded: frames={dataset.num_frames}, episodes={dataset.num_episodes}, fps={fps}")

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
    _log(f"DataLoader ready: batches_per_epoch={len(loader)}, workers={args.num_workers}")

    feature_type = {
        "VISUAL": FeatureType.VISUAL,
        "STATE": FeatureType.STATE,
        "ACTION": FeatureType.ACTION,
    }
    input_features = _to_policy_features(adapter.lerobot_input_features(), PolicyFeature, feature_type)
    output_features = _to_policy_features(adapter.lerobot_output_features(), PolicyFeature, feature_type)

    cfg_cls = PI05Config if args.policy_type == "pi05" else PI0Config
    policy_cls = PI05Policy if args.policy_type == "pi05" else PI0Policy
    cfg = cfg_cls(
        input_features=input_features,
        output_features=output_features,
        device=args.device,
        dtype=args.dtype,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        max_state_dim=args.max_state_dim,
        max_action_dim=args.max_action_dim,
        image_resolution=(args.image_size, args.image_size),
        num_inference_steps=args.num_inference_steps,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        freeze_vision_encoder=args.freeze_vision_encoder,
        train_expert_only=args.train_expert_only,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=args.compile_model,
        optimizer_lr=args.lr,
        optimizer_betas=(args.optimizer_beta1, args.optimizer_beta2),
        optimizer_eps=args.optimizer_eps,
        optimizer_weight_decay=args.weight_decay,
        optimizer_grad_clip_norm=args.grad_clip_norm,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_decay_steps=args.scheduler_decay_steps,
        scheduler_decay_lr=args.scheduler_decay_lr,
    )
    _log(f"Policy config built: policy_type={args.policy_type}")

    _log(
        "Initializing PI policy model; default pi0 uses a large PaliGemma backbone "
        "and this step can take several minutes on CPU/GPU"
    )
    policy = _make_policy(
        policy_cls=policy_cls,
        cfg=cfg,
        pretrained_path=args.pretrained_path,
        local_files_only=args.local_files_only,
    )
    policy.to(args.device)
    policy.train()
    total_params, trainable_params = _count_parameters(policy)
    _log(
        "Policy initialized and moved to device: "
        f"total_params={total_params:,}, trainable_params={trainable_params:,}"
    )
    _log("Building PI preprocessor/tokenizer; this may access google/paligemma-3b-pt-224 from HF cache/hub")
    _patch_paligemma_tokenizer(args.paligemma_tokenizer_path)
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, dataset_stats=dataset_stats)
    _log("Preprocessor/tokenizer ready")

    optim_params = _trainable_optim_params(policy.get_optim_params())
    optim_param_count = sum(param.numel() for param in optim_params)
    _log(
        "Optimizer params prepared: "
        f"tensors={len(optim_params):,}, params={int(optim_param_count):,}"
    )
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        betas=(args.optimizer_beta1, args.optimizer_beta2),
        eps=args.optimizer_eps,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_cosine_warmup_decay_lambda(
            total_steps=args.steps,
            warmup_steps=args.scheduler_warmup_steps,
            decay_steps=args.scheduler_decay_steps,
            final_lr_ratio=args.scheduler_decay_lr / args.lr,
        ),
    )

    stats_path = output_dir / "dataset_stats.json"
    run_config = {
        "policy_type": args.policy_type,
        "dataset_root": str(dataset_root),
        "dataset_frames": int(dataset.num_frames),
        "dataset_episodes": int(dataset.num_episodes),
        "fps": fps,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "chunk_size": int(args.chunk_size),
        "n_action_steps": int(args.n_action_steps),
        "image_shape_chw": list(adapter.spec().image_shape_chw),
        "max_state_dim": int(args.max_state_dim),
        "max_action_dim": int(args.max_action_dim),
        "input_feature_keys": list(cfg.input_features.keys()),
        "output_feature_keys": list(cfg.output_features.keys()),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "num_inference_steps": int(args.num_inference_steps),
        "paligemma_variant": args.paligemma_variant,
        "action_expert_variant": args.action_expert_variant,
        "freeze_vision_encoder": bool(args.freeze_vision_encoder),
        "train_expert_only": bool(args.train_expert_only),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "compile_model": bool(args.compile_model),
        "pretrained_path": args.pretrained_path,
        "paligemma_tokenizer_path": args.paligemma_tokenizer_path,
        "task_override": args.task,
        "dataset_stats_path": str(stats_path),
    }
    _write_json(output_dir / "run_config.json", run_config)
    _write_json(stats_path, dataset_stats)

    use_wandb = args.wandb
    if use_wandb:
        import wandb

        wandb_kwargs: dict[str, Any] = dict(project=args.wandb_project, config=run_config, dir=str(output_dir))
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_name:
            wandb_kwargs["name"] = args.wandb_name
        if args.wandb_offline:
            wandb_kwargs["mode"] = "offline"
        wandb.init(**wandb_kwargs)

    log_path = output_dir / "train_log.jsonl"
    first_shapes: dict[str, list[int]] | None = None
    step = 0
    data_iter = iter(loader)
    start = time.perf_counter()
    last_loss = math.nan

    with log_path.open("w", encoding="utf-8") as log_f:
        while step < args.steps:
            try:
                if step == 0:
                    _log("Fetching first DataLoader batch")
                raw_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                raw_batch = next(data_iter)

            step += 1
            step_start = time.perf_counter()
            batch = adapter.to_training_batch(raw_batch, device=None, task_override=args.task)
            batch = preprocessor(batch)
            if step == 1:
                _log("First batch preprocessed; running first forward/backward")
            if first_shapes is None:
                first_shapes = {key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")}

            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy.forward(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
            optimizer.step()
            lr_scheduler.step()

            last_loss = float(loss.detach().cpu())
            entry = {
                "step": step,
                "loss": last_loss,
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "step_s": time.perf_counter() - step_start,
                "loss_dict": _jsonable_loss_dict(loss_dict),
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
                    lr_scheduler,
                    cfg,
                    step,
                    run_config,
                    dataset_stats,
                )

    final_ckpt = output_dir / "checkpoint_last.pt"
    _save_checkpoint(final_ckpt, policy, optimizer, lr_scheduler, cfg, step, run_config, dataset_stats)
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


def _make_policy(policy_cls: Any, cfg: Any, pretrained_path: str | None, local_files_only: bool) -> torch.nn.Module:
    if pretrained_path:
        return policy_cls.from_pretrained(
            pretrained_path,
            config=cfg,
            strict=False,
            local_files_only=local_files_only,
        )
    return policy_cls(cfg)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return int(total), int(trainable)


def _trainable_optim_params(params: Any) -> list[torch.nn.Parameter]:
    trainable = [param for param in params if param.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found for optimizer")
    return trainable


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


def _require_quantile_stats(stats: dict[str, Any]) -> None:
    missing = [
        key
        for key in (STATE_KEY, ACTION_KEY)
        if key not in stats or "q01" not in stats[key] or "q99" not in stats[key]
    ]
    if missing:
        raise SystemExit(
            "PI0.5 uses QUANTILES normalization and needs q01/q99 stats for "
            f"{missing}. Run LeRobot's augment_dataset_quantile_stats.py for this dataset, "
            "or train with --policy-type pi0 which uses mean/std stats."
        )


def _require_mean_std_stats(stats: dict[str, Any]) -> None:
    missing = [
        key
        for key in (STATE_KEY, ACTION_KEY)
        if key not in stats or "mean" not in stats[key] or "std" not in stats[key]
    ]
    if missing:
        raise SystemExit(f"PI0 uses MEAN_STD normalization and needs mean/std stats for {missing}.")


def _cosine_warmup_decay_lambda(
    *,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    final_lr_ratio: float,
) -> Any:
    warmup_steps = max(0, int(warmup_steps))
    decay_steps = max(1, min(int(decay_steps), int(total_steps)))
    final_lr_ratio = max(0.0, float(final_lr_ratio))

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / float(max(1, decay_steps - warmup_steps))))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine

    return fn


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


def _configure_hf_env(args: Any) -> None:
    cache_root = Path(args.hf_home).expanduser().resolve() if args.hf_home else Path.cwd() / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = str(args.hf_endpoint)
    if args.hf_offline or args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for key in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def _log(message: str) -> None:
    print(f"[lerobot-pi-train] {message}", flush=True)


def _patch_paligemma_tokenizer(local_path: str | None) -> None:
    if not local_path:
        return
    path = Path(local_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"--paligemma-tokenizer-path not found: {path}")

    from transformers import AutoTokenizer

    original_from_pretrained = AutoTokenizer.from_pretrained

    def redirected_from_pretrained(pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(pretrained_model_name_or_path) == "google/paligemma-3b-pt-224":
            pretrained_model_name_or_path = str(path)
            kwargs.setdefault("local_files_only", True)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    AutoTokenizer.from_pretrained = staticmethod(redirected_from_pretrained)  # type: ignore[method-assign]
    _log(f"Redirecting google/paligemma-3b-pt-224 tokenizer to local path: {path}")


def _save_checkpoint(
    path: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
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
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "pi_config": _jsonable_config(cfg),
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


def _jsonable_loss_dict(loss_dict: dict[str, Any] | None) -> dict[str, Any]:
    if not loss_dict:
        return {}
    return {str(key): _jsonable(value) for key, value in loss_dict.items()}


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
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value)
        return value.numpy().tolist()
    if hasattr(value, "type") and hasattr(value, "shape"):
        return {"type": str(value.type), "shape": list(value.shape)}
    return str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

