from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import safetensors.torch
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_SOURCE_DIR = PROJECT_DIR / "source" / "whole_body_tracking"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_SOURCE_DIR))

from isaaclab.app import AppLauncher


def load_object(entry_point: str):
    import importlib

    module_name, object_name = entry_point.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), object_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug G1 tracking eval parallel motion assignment through EvalHook.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_DIR / "logs" / "tmp_fbcpr_g1" / "G1FBCPR" / "checkpoint"),
        help="FB-CPR checkpoint folder or model folder. Defaults to logs/tmp_fbcpr_g1/G1FBCPR/checkpoint.",
    )
    parser.add_argument("--zero-agent", action="store_true", help="Use a zero-action dummy model instead of loading a checkpoint.")
    parser.add_argument("--motions", type=str, default=str(PROJECT_SOURCE_DIR / "bfm" / "data" / "lafan"))
    parser.add_argument("--num-envs", type=int, default=8, help="Total IsaacLab envs to create.")
    parser.add_argument("--tracking-eval-num-envs", type=int, default=4, help="Vectorized envs used by one eval chunk.")
    parser.add_argument("--tracking-eval-max-motions", type=int, default=4, help="Number of motions to evaluate.")
    parser.add_argument("--tracking-eval-max-steps", type=int, default=4, help="Rollout steps per eval chunk.")
    parser.add_argument("--tracking-eval-progress", action="store_true", help="Show the evaluator tqdm bar.")
    parser.add_argument("--assignment-limit", type=int, default=32, help="Maximum env->motion rows to print per chunk.")
    parser.add_argument("--hook-step", type=int, default=1, help="Runner step value used to trigger EvalHook.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--debug-vis", action="store_true", help="Show expert body markers if running with a viewer.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ZeroTrackingModel:
    def __init__(self, obs_dim: int, action_dim: int, z_dim: int, device: str):
        self.cfg = SimpleNamespace(obs_dim=obs_dim, action_dim=action_dim, z_dim=z_dim, device=device)
        self.training = True

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def tracking_inference(self, reference_obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(reference_obs.shape[0], self.cfg.z_dim, dtype=reference_obs.dtype, device=reference_obs.device)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        del z, mean
        return torch.zeros(obs.shape[0], self.cfg.action_dim, dtype=obs.dtype, device=obs.device)


class ZeroAgent:
    def __init__(self, obs_dim: int, action_dim: int, z_dim: int, device: str):
        self._model = ZeroTrackingModel(obs_dim=obs_dim, action_dim=action_dim, z_dim=z_dim, device=device)


class ModelAgent:
    def __init__(self, model):
        self._model = model


class HookDebugRunner:
    def __init__(self, env, step: int):
        self.env = env
        self.step = step
        self.num_env_steps = step
        self.observation_refresh_requested = False

    def request_observation_refresh(self) -> None:
        self.observation_refresh_requested = True
        print("[HookDebugRunner] request_observation_refresh() called", flush=True)


def _resolve_model_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint path {path!s}.")
    return path / "model" if (path / "model").is_dir() else path


def _load_fbcpr_model(checkpoint: str | Path, device: str):
    from agents.metamotivo.fb_cpr.model import FBcprModel

    model_dir = _resolve_model_dir(checkpoint)
    with (model_dir / "config.json").open() as f:
        model_cfg = json.load(f)
    model_cfg["device"] = device
    model = FBcprModel(**model_cfg)
    missing, unexpected = safetensors.torch.load_model(
        model,
        model_dir / "model.safetensors",
        strict=False,
        device=device,
    )
    if missing:
        print(f"[INFO] loaded checkpoint with missing_keys={len(missing)}", flush=True)
    if unexpected:
        print(
            f"[INFO] ignored {len(unexpected)} training-only checkpoint keys while loading eval model",
            flush=True,
        )
    model.train(False)
    print(f"[INFO] loaded FB-CPR model from {model_dir}", flush=True)
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    print(
        "[INFO] launching IsaacLab for G1 eval parallel debug "
        f"device={args.device} headless={args.headless}",
        flush=True,
    )
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print("[INFO] IsaacLab app launched", flush=True)

    raw_env = None
    try:
        print("[INFO] importing gymnasium", flush=True)
        import gymnasium as gym

        print("[INFO] importing bfm task registry", flush=True)
        import bfm.tasks  # noqa: F401

        print("[INFO] importing runner/eval helpers", flush=True)
        from agents.runner.env import IsaacLabVecEnvAdapter, unpack_observations
        from agents.runner.env_runner import EvalHook
        from bfm.config.fb_cpr import G1FBcprRunnerCfg

        cfg = G1FBcprRunnerCfg()
        cfg.seed = args.seed
        cfg.motions = args.motions
        cfg.online_parallel_envs = args.num_envs
        cfg.device = args.device
        cfg.evaluate = True
        cfg.eval_every_steps = args.hook_step
        cfg.tracking_eval_num_envs = args.tracking_eval_num_envs
        cfg.tracking_eval_max_motions = args.tracking_eval_max_motions
        cfg.tracking_eval_max_steps = args.tracking_eval_max_steps
        cfg.tracking_eval_progress = args.tracking_eval_progress
        cfg.tracking_eval_print_assignments = True
        cfg.tracking_eval_print_assignment_limit = args.assignment_limit
        cfg.__post_init__()

        env_cfg = load_object(cfg.env_cfg_entry_point)()
        configure_env_cfg = load_object(cfg.configure_env_entry_point)
        env_cfg = configure_env_cfg(env_cfg=env_cfg, cfg=cfg, args_cli=args)
        env_cfg.commands.motion.debug_vis = args.debug_vis
        print(
            f"[INFO] creating env_id={cfg.env_id} num_envs={cfg.online_parallel_envs} motions={cfg.motions}",
            flush=True,
        )
        raw_env = gym.make(cfg.env_id, cfg=env_cfg)
        vec_env = IsaacLabVecEnvAdapter(raw_env)
        print("[INFO] resetting env", flush=True)
        obs_raw, _ = raw_env.reset()
        obs, _ = unpack_observations(obs_raw)
        obs_dim = int(obs.shape[1])
        action_dim = int(raw_env.unwrapped.action_manager.total_action_dim)
        print(f"[INFO] env ready obs_dim={obs_dim} action_dim={action_dim}", flush=True)

        if args.zero_agent:
            agent = ZeroAgent(obs_dim=obs_dim, action_dim=action_dim, z_dim=args.z_dim, device=args.device)
            print("[INFO] using zero-action debug agent", flush=True)
        else:
            model = _load_fbcpr_model(args.checkpoint, device=args.device)
            if model.cfg.obs_dim != obs_dim or model.cfg.action_dim != action_dim:
                raise ValueError(
                    f"Checkpoint dims obs/action={model.cfg.obs_dim}/{model.cfg.action_dim} "
                    f"do not match env dims {obs_dim}/{action_dim}."
                )
            agent = ModelAgent(model)
        evaluator_factory = load_object(cfg.tracking_eval_entry_point)
        evaluator = evaluator_factory(cfg=cfg, agent=agent, env=raw_env)
        hook = EvalHook(every_steps=args.hook_step, enabled=True, use_wandb=False, evaluator=evaluator)
        runner = HookDebugRunner(env=vec_env, step=args.hook_step)

        print(
            "[INFO] calling EvalHook with "
            f"num_envs={args.num_envs} eval_envs={args.tracking_eval_num_envs} "
            f"max_motions={args.tracking_eval_max_motions} max_steps={args.tracking_eval_max_steps}",
            flush=True,
        )
        hook.on_train_start(runner)
        hook.on_step_end(runner, transition=None)
        hook.on_train_end(runner)
        print(
            f"[INFO] done observation_refresh_requested={runner.observation_refresh_requested}",
            flush=True,
        )
    except BaseException as exc:
        print(f"[ERROR] debug_g1_eval_parallel failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        if raw_env is not None:
            raw_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
