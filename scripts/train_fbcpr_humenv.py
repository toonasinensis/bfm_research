from __future__ import annotations

import dataclasses
import json
import random
import sys
import argparse
import uuid
from dataclasses import field
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_SOURCE_DIR = PROJECT_DIR / "source" / "whole_body_tracking"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_SOURCE_DIR))

from isaaclab.app import AppLauncher


@dataclasses.dataclass
class TrainConfig:
    seed: int = 0
    motions: str = ""
    motions_root: str = ""
    buffer_size: int = 5_000_000
    online_parallel_envs: int = 50
    log_every_updates: int = 100_000
    work_dir: str | None = None
    num_env_steps: int = 30_000_000
    update_agent_every: int | None = None
    num_seed_steps: int | None = None
    num_agent_updates: int | None = None
    checkpoint_every_steps: int = 5_000_000
    prioritization: bool = False

    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = "fbcpr_humenv"

    compile: bool = False
    cudagraphs: bool = False
    device: str = "cuda"
    buffer_device: str = "cpu"

    evaluate: bool = False
    eval_every_steps: int = 1_000_000

    model: str = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 2
    z_dim: int = 256
    seq_length: int = 8
    actor_std: float = 0.2
    batch_size: int = 1024
    reward_eval_tasks: list[str] = field(default_factory=list)
    motion_reset: bool = True
    fall_prob: float = 0.2
    fall_height: float = 0.35

    def __post_init__(self):
        if self.update_agent_every is None:
            self.update_agent_every = 10 * self.online_parallel_envs
        if self.num_seed_steps is None:
            self.num_seed_steps = 1000 * self.online_parallel_envs
        if self.num_agent_updates is None:
            self.num_agent_updates = self.online_parallel_envs
        if self.prioritization:
            raise NotImplementedError("FB-CPR prioritization is disabled in this port.")


def _add_optional_int(parser: argparse.ArgumentParser, name: str, default: int | None, help_text: str) -> None:
    parser.add_argument(name, type=int, default=default, help=help_text)


def parse_args() -> tuple[argparse.Namespace, TrainConfig]:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Train FB-CPR on the IsaacLab HumEnv MJCF task.")
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--motions", type=str, default=cfg.motions)
    parser.add_argument("--motions-root", type=str, default=cfg.motions_root)
    parser.add_argument("--buffer-size", type=int, default=cfg.buffer_size)
    parser.add_argument("--online-parallel-envs", type=int, default=cfg.online_parallel_envs)
    parser.add_argument("--log-every-updates", type=int, default=cfg.log_every_updates)
    parser.add_argument("--work-dir", type=str, default=cfg.work_dir)
    parser.add_argument("--num-env-steps", type=int, default=cfg.num_env_steps)
    _add_optional_int(parser, "--update-agent-every", cfg.update_agent_every, "Agent update interval in env steps.")
    _add_optional_int(parser, "--num-seed-steps", cfg.num_seed_steps, "Random-action seed steps.")
    _add_optional_int(parser, "--num-agent-updates", cfg.num_agent_updates, "Agent updates per update event.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=cfg.checkpoint_every_steps)
    parser.add_argument("--prioritization", action="store_true", default=cfg.prioritization)
    parser.add_argument("--use-wandb", action="store_true", default=cfg.use_wandb)
    parser.add_argument("--wandb-ename", type=str, default=cfg.wandb_ename)
    parser.add_argument("--wandb-gname", type=str, default=cfg.wandb_gname)
    parser.add_argument("--wandb-pname", type=str, default=cfg.wandb_pname)
    parser.add_argument("--compile", action="store_true", default=cfg.compile)
    parser.add_argument("--cudagraphs", action="store_true", default=cfg.cudagraphs)
    parser.add_argument("--agent-device", type=str, default=None, help="Torch device for FB-CPR. Defaults to launcher device.")
    parser.add_argument("--buffer-device", type=str, default=cfg.buffer_device)
    parser.add_argument("--evaluate", action="store_true", default=cfg.evaluate)
    parser.add_argument("--eval-every-steps", type=int, default=cfg.eval_every_steps)
    parser.add_argument("--model", type=str, default=cfg.model)
    parser.add_argument("--hidden-dim", type=int, default=cfg.hidden_dim)
    parser.add_argument("--hidden-layers", type=int, default=cfg.hidden_layers)
    parser.add_argument("--z-dim", type=int, default=cfg.z_dim)
    parser.add_argument("--seq-length", type=int, default=cfg.seq_length)
    parser.add_argument("--actor-std", type=float, default=cfg.actor_std)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument(
        "--no-motion-reset",
        action="store_false",
        dest="motion_reset",
        default=cfg.motion_reset,
        help="Disable MoCap/Fall state reset from HumEnv motion qpos/qvel.",
    )
    parser.add_argument("--fall-prob", type=float, default=cfg.fall_prob)
    parser.add_argument("--fall-height", type=float, default=cfg.fall_height)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    cfg.seed = args.seed
    cfg.motions = args.motions
    cfg.motions_root = args.motions_root
    cfg.buffer_size = args.buffer_size
    cfg.online_parallel_envs = args.online_parallel_envs
    cfg.log_every_updates = args.log_every_updates
    cfg.work_dir = args.work_dir
    cfg.num_env_steps = args.num_env_steps
    cfg.update_agent_every = args.update_agent_every
    cfg.num_seed_steps = args.num_seed_steps
    cfg.num_agent_updates = args.num_agent_updates
    cfg.checkpoint_every_steps = args.checkpoint_every_steps
    cfg.prioritization = args.prioritization
    cfg.use_wandb = args.use_wandb
    cfg.wandb_ename = args.wandb_ename
    cfg.wandb_gname = args.wandb_gname
    cfg.wandb_pname = args.wandb_pname
    cfg.compile = args.compile
    cfg.cudagraphs = args.cudagraphs
    cfg.device = args.agent_device or args.device
    cfg.buffer_device = args.buffer_device
    cfg.evaluate = args.evaluate
    cfg.eval_every_steps = args.eval_every_steps
    cfg.model = args.model
    cfg.hidden_dim = args.hidden_dim
    cfg.hidden_layers = args.hidden_layers
    cfg.z_dim = args.z_dim
    cfg.seq_length = args.seq_length
    cfg.actor_std = args.actor_std
    cfg.batch_size = args.batch_size
    cfg.motion_reset = args.motion_reset
    cfg.fall_prob = args.fall_prob
    cfg.fall_height = args.fall_height
    cfg.__post_init__()
    return args, cfg


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_work_dir(cfg: TrainConfig) -> Path:
    if cfg.work_dir is None:
        tmp_name = uuid.uuid4().hex[:10].upper()
        work_dir = Path.cwd() / "tmp_fbcpr" / tmp_name
        cfg.work_dir = str(work_dir)
    else:
        work_dir = Path(cfg.work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)
    return work_dir


def main() -> None:
    args_cli, cfg = parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym

    try:
        import wandb
    except ImportError:  # pragma: no cover
        wandb = None

    import bfm.tasks  # noqa: F401
    from agents.metamotivo.buffers.buffers import DictBuffer
    from agents.metamotivo.fb_cpr import FBcprAgent
    from agents.runner.env import IsaacLabVecEnvAdapter
    from agents.runner.env_runner import CheckpointHook, EnvRunner, EvalHook, FBcprAdapter, LogHook
    from bfm.tasks.humenv.config.fb_cpr import build_fbcpr_agent_config
    from bfm.tasks.humenv.humenv_env_cfg import HumEnvMjcfEnvCfg
    from bfm.tasks.humenv.mdp.commands import load_expert_trajectories

    set_seed_everywhere(cfg.seed)
    work_dir = make_work_dir(cfg)
    print(f"Workdir: {work_dir}", flush=True)

    with (work_dir / "config.json").open("w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)

    if cfg.use_wandb:
        if wandb is None:
            raise ImportError("wandb is required when use_wandb=True.")
        try:
            wandb.init(
                entity=cfg.wandb_ename,
                project=cfg.wandb_pname,
                group=cfg.wandb_gname,
                name=f"fbcpr-{work_dir.name}",
                config=dataclasses.asdict(cfg),
                anonymous="allow",
            )
        except Exception as exc:
            print(f"[WARN] wandb init failed, continuing without online logging: {exc}", flush=True)
            cfg.use_wandb = False

    env_cfg = HumEnvMjcfEnvCfg()
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.online_parallel_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.commands.motion.motions = cfg.motions
    env_cfg.commands.motion.motions_root = cfg.motions_root
    env_cfg.commands.motion.reset_robot_state = cfg.motion_reset
    env_cfg.commands.motion.fall_prob = cfg.fall_prob
    env_cfg.commands.motion.fall_height = cfg.fall_height

    raw_env = gym.make("HumEnv-MJCF-Flat-v0", cfg=env_cfg)
    env = IsaacLabVecEnvAdapter(raw_env)
    obs, _ = env.reset()
    obs_dim = obs.shape[1]
    action_dim = env.num_actions

    agent_cfg = build_fbcpr_agent_config(obs_dim=obs_dim, action_dim=action_dim, cfg=cfg)
    agent = FBcprAgent(**dataclasses.asdict(agent_cfg))

    print("Loading expert trajectories", flush=True)
    expert_buffer = load_expert_trajectories(
        cfg.motions,
        cfg.motions_root,
        device=cfg.buffer_device,
        sequence_length=agent_cfg.model.seq_length,
    )
    replay_buffer = {
        "train": DictBuffer(capacity=cfg.buffer_size, device=cfg.buffer_device),
        "expert_slicer": expert_buffer,
    }

    adapter = FBcprAdapter(
        agent=agent,
        replay_buffer=replay_buffer,
        num_seed_steps=cfg.num_seed_steps,
        update_agent_every=cfg.update_agent_every,
        num_agent_updates=cfg.num_agent_updates,
        action_dim=action_dim,
        action_device=env.device,
    )
    hooks = [
        EvalHook(every_steps=cfg.eval_every_steps, enabled=cfg.evaluate),
        LogHook(every_steps=cfg.log_every_updates, use_wandb=cfg.use_wandb),
        CheckpointHook(every_steps=cfg.checkpoint_every_steps, output_dir=work_dir),
    ]

    try:
        runner = EnvRunner(env=env, algorithm=adapter, hooks=hooks)
        runner.train(num_env_steps=cfg.num_env_steps)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
