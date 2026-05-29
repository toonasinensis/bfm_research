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
    task: str = "humenv"
    runner_cfg_entry_point: str = "bfm.config.fb_cpr:HumEnvFBcprRunnerCfg"
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
    wandb_pname: str | None = None
    progress: bool = True

    compile: bool = False
    cudagraphs: bool = False
    device: str = "cuda"
    buffer_device: str = "cpu"

    env_id: str = "HumEnv-MJCF-Flat-v0"
    env_cfg_entry_point: str = "bfm.tasks.humenv.humenv_env_cfg:HumEnvMjcfEnvCfg"
    configure_env_entry_point: str = "bfm.tasks.humenv.config.fb_cpr:configure_env_cfg"
    expert_buffer_entry_point: str = "bfm.tasks.humenv.config.fb_cpr:make_expert_buffer"
    agent_class_entry_point: str = "agents.metamotivo.fb_cpr:FBcprAgent"
    agent_config_builder_entry_point: str = "bfm.config.fb_cpr:build_fbcpr_agent_config"

    evaluate: bool = False
    eval_every_steps: int = 1_000_000
    tracking_eval_entry_point: str = "bfm.tasks.humenv.eval.fbcpr_tracking:make_tracking_evaluator"
    tracking_eval_num_envs: int = 1
    tracking_eval_motions: str = ""
    tracking_eval_motions_root: str = ""
    tracking_eval_max_motions: int = 0
    tracking_eval_max_steps: int = 0
    tracking_eval_mean_action: bool = True
    tracking_eval_progress: bool = True

    model: str = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 2
    z_dim: int = 256
    seq_length: int = 8
    actor_std: float = 0.2
    batch_size: int = 1024
    discount: float = 0.98
    update_z_every_step: int = 150
    lr_f: float = 1e-4
    lr_b: float = 1e-5
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    lr_aux_critic: float = 1e-4
    lr_discriminator: float = 1e-5
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    fb_target_tau: float = 0.01
    critic_target_tau: float = 0.005
    ortho_coef: float = 100.0
    train_goal_ratio: float = 0.2
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    critic_pessimism_penalty: float = 0.5
    aux_critic_pessimism_penalty: float = 0.5
    stddev_clip: float = 0.3
    expert_asm_ratio: float = 0.6
    relabel_ratio: float = 0.8
    reg_coeff: float = 0.01
    reg_coeff_aux: float = 1.0
    scale_reg: bool = True
    q_loss_coef: float = 0.1
    grad_penalty_discriminator: float = 10.0
    weight_decay_discriminator: float = 0.0
    use_mix_rollout: bool = True
    z_buffer_size: int = 10000
    inference_batch_size: int = 500_000
    norm_obs: bool = True
    norm_z: bool = True
    b_norm: bool = True
    b_hidden_dim: int = 256
    b_hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: str = "batch"
    discriminator_hidden_dim: int = 1024
    discriminator_hidden_layers: int = 3
    norm_aux_reward_translate: bool = False
    norm_aux_reward_scale: bool = True
    aux_rewards: list[str] = field(default_factory=list)
    aux_rewards_scaling: dict[str, float] = field(default_factory=dict)
    reward_eval_tasks: list[str] = field(default_factory=list)
    motion_reset: bool = True
    fall_prob: float = 0.2
    fall_height: float = 0.35

    def __post_init__(self):
        if self.wandb_pname is None:
            self.wandb_pname = f"fbcpr_{self.task}"
        if self.update_agent_every is None:
            self.update_agent_every = 10 * self.online_parallel_envs
        if self.num_seed_steps is None:
            self.num_seed_steps = 1000 * self.online_parallel_envs
        if self.num_agent_updates is None:
            self.num_agent_updates = self.online_parallel_envs
        if self.prioritization:
            raise NotImplementedError("FB-CPR prioritization is disabled in this port.")


TASK_DEFAULT_CFG_ENTRY_POINTS = {
    "humenv": "bfm.config.fb_cpr:HumEnvFBcprRunnerCfg",
    "g1": "bfm.config.fb_cpr:G1FBcprRunnerCfg",
}


def load_object(entry_point: str):
    import importlib

    module_name, object_name = entry_point.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), object_name)


def get_cli_task(default: str = "humenv") -> str:
    for index, arg in enumerate(sys.argv):
        if arg == "--task" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith("--task="):
            return arg.split("=", maxsplit=1)[1]
    return default


def get_cli_runner_cfg_entry_point() -> str | None:
    for index, arg in enumerate(sys.argv):
        if arg == "--runner-cfg-entry-point" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith("--runner-cfg-entry-point="):
            return arg.split("=", maxsplit=1)[1]
    return None


def make_default_train_config(task: str, entry_point: str | None = None) -> TrainConfig:
    selected_entry_point = entry_point or TASK_DEFAULT_CFG_ENTRY_POINTS.get(task)
    cfg = TrainConfig(task=task)
    if selected_entry_point is None:
        cfg.__post_init__()
        return cfg

    task_cfg = load_object(selected_entry_point)()
    train_fields = {cfg_field.name for cfg_field in dataclasses.fields(TrainConfig)}
    for name in train_fields:
        if hasattr(task_cfg, name):
            setattr(cfg, name, getattr(task_cfg, name))
    cfg.task = getattr(task_cfg, "task", task)
    cfg.runner_cfg_entry_point = selected_entry_point
    cfg.__post_init__()
    return cfg


def _add_optional_int(parser: argparse.ArgumentParser, name: str, default: int | None, help_text: str) -> None:
    parser.add_argument(name, type=int, default=default, help=help_text)


def parse_args() -> tuple[argparse.Namespace, TrainConfig]:
    cfg = make_default_train_config(get_cli_task(), entry_point=get_cli_runner_cfg_entry_point())
    parser = argparse.ArgumentParser(description="Train FB-CPR on IsaacLab HumEnv or G1 LAFAN tasks.")
    parser.add_argument("--task", type=str, default=cfg.task)
    parser.add_argument("--runner-cfg-entry-point", type=str, default=cfg.runner_cfg_entry_point)
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
    parser.add_argument("--no-progress", action="store_false", dest="progress", default=cfg.progress)
    parser.add_argument("--compile", action="store_true", default=cfg.compile)
    parser.add_argument("--cudagraphs", action="store_true", default=cfg.cudagraphs)
    parser.add_argument("--agent-device", type=str, default=None, help="Torch device for FB-CPR. Defaults to launcher device.")
    parser.add_argument("--buffer-device", type=str, default=cfg.buffer_device)
    parser.add_argument("--env-id", type=str, default=cfg.env_id)
    parser.add_argument("--env-cfg-entry-point", type=str, default=cfg.env_cfg_entry_point)
    parser.add_argument("--configure-env-entry-point", type=str, default=cfg.configure_env_entry_point)
    parser.add_argument("--expert-buffer-entry-point", type=str, default=cfg.expert_buffer_entry_point)
    parser.add_argument("--agent-class-entry-point", type=str, default=cfg.agent_class_entry_point)
    parser.add_argument("--agent-config-builder-entry-point", type=str, default=cfg.agent_config_builder_entry_point)
    parser.add_argument("--evaluate", action="store_true", default=cfg.evaluate)
    parser.add_argument("--eval-every-steps", type=int, default=cfg.eval_every_steps)
    parser.add_argument("--tracking-eval-entry-point", type=str, default=cfg.tracking_eval_entry_point)
    parser.add_argument("--tracking-eval-num-envs", type=int, default=cfg.tracking_eval_num_envs)
    parser.add_argument("--tracking-eval-motions", type=str, default=cfg.tracking_eval_motions)
    parser.add_argument("--tracking-eval-motions-root", type=str, default=cfg.tracking_eval_motions_root)
    parser.add_argument("--tracking-eval-max-motions", type=int, default=cfg.tracking_eval_max_motions)
    parser.add_argument("--tracking-eval-max-steps", type=int, default=cfg.tracking_eval_max_steps)
    parser.add_argument("--tracking-eval-sample-action", action="store_false", dest="tracking_eval_mean_action", default=cfg.tracking_eval_mean_action)
    parser.add_argument("--no-tracking-eval-progress", action="store_false", dest="tracking_eval_progress", default=cfg.tracking_eval_progress)
    parser.add_argument("--model", type=str, default=cfg.model)
    parser.add_argument("--hidden-dim", type=int, default=cfg.hidden_dim)
    parser.add_argument("--hidden-layers", type=int, default=cfg.hidden_layers)
    parser.add_argument("--z-dim", type=int, default=cfg.z_dim)
    parser.add_argument("--seq-length", type=int, default=cfg.seq_length)
    parser.add_argument("--actor-std", type=float, default=cfg.actor_std)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--discount", type=float, default=cfg.discount)
    parser.add_argument("--update-z-every-step", type=int, default=cfg.update_z_every_step)
    parser.add_argument("--lr-f", type=float, default=cfg.lr_f)
    parser.add_argument("--lr-b", type=float, default=cfg.lr_b)
    parser.add_argument("--lr-actor", type=float, default=cfg.lr_actor)
    parser.add_argument("--lr-critic", type=float, default=cfg.lr_critic)
    parser.add_argument("--lr-aux-critic", type=float, default=cfg.lr_aux_critic)
    parser.add_argument("--lr-discriminator", type=float, default=cfg.lr_discriminator)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--clip-grad-norm", type=float, default=cfg.clip_grad_norm)
    parser.add_argument("--fb-target-tau", type=float, default=cfg.fb_target_tau)
    parser.add_argument("--critic-target-tau", type=float, default=cfg.critic_target_tau)
    parser.add_argument("--ortho-coef", type=float, default=cfg.ortho_coef)
    parser.add_argument("--train-goal-ratio", type=float, default=cfg.train_goal_ratio)
    parser.add_argument("--fb-pessimism-penalty", type=float, default=cfg.fb_pessimism_penalty)
    parser.add_argument("--actor-pessimism-penalty", type=float, default=cfg.actor_pessimism_penalty)
    parser.add_argument("--critic-pessimism-penalty", type=float, default=cfg.critic_pessimism_penalty)
    parser.add_argument("--aux-critic-pessimism-penalty", type=float, default=cfg.aux_critic_pessimism_penalty)
    parser.add_argument("--stddev-clip", type=float, default=cfg.stddev_clip)
    parser.add_argument("--expert-asm-ratio", type=float, default=cfg.expert_asm_ratio)
    parser.add_argument("--relabel-ratio", type=float, default=cfg.relabel_ratio)
    parser.add_argument("--reg-coeff", type=float, default=cfg.reg_coeff)
    parser.add_argument("--reg-coeff-aux", type=float, default=cfg.reg_coeff_aux)
    parser.add_argument("--no-scale-reg", action="store_false", dest="scale_reg", default=cfg.scale_reg)
    parser.add_argument("--q-loss-coef", type=float, default=cfg.q_loss_coef)
    parser.add_argument("--grad-penalty-discriminator", type=float, default=cfg.grad_penalty_discriminator)
    parser.add_argument("--weight-decay-discriminator", type=float, default=cfg.weight_decay_discriminator)
    parser.add_argument("--no-use-mix-rollout", action="store_false", dest="use_mix_rollout", default=cfg.use_mix_rollout)
    parser.add_argument("--z-buffer-size", type=int, default=cfg.z_buffer_size)
    parser.add_argument("--inference-batch-size", type=int, default=cfg.inference_batch_size)
    parser.add_argument("--no-norm-obs", action="store_false", dest="norm_obs", default=cfg.norm_obs)
    parser.add_argument("--no-norm-z", action="store_false", dest="norm_z", default=cfg.norm_z)
    parser.add_argument("--no-b-norm", action="store_false", dest="b_norm", default=cfg.b_norm)
    parser.add_argument("--b-hidden-dim", type=int, default=cfg.b_hidden_dim)
    parser.add_argument("--b-hidden-layers", type=int, default=cfg.b_hidden_layers)
    parser.add_argument("--embedding-layers", type=int, default=cfg.embedding_layers)
    parser.add_argument("--num-parallel", type=int, default=cfg.num_parallel)
    parser.add_argument("--ensemble-mode", type=str, choices=["batch", "seq", "vmap"], default=cfg.ensemble_mode)
    parser.add_argument("--discriminator-hidden-dim", type=int, default=cfg.discriminator_hidden_dim)
    parser.add_argument("--discriminator-hidden-layers", type=int, default=cfg.discriminator_hidden_layers)
    parser.add_argument("--norm-aux-reward-translate", action="store_true", default=cfg.norm_aux_reward_translate)
    parser.add_argument("--no-norm-aux-reward-scale", action="store_false", dest="norm_aux_reward_scale", default=cfg.norm_aux_reward_scale)
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

    cfg.task = args.task
    cfg.runner_cfg_entry_point = args.runner_cfg_entry_point
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
    cfg.progress = args.progress
    cfg.compile = args.compile
    cfg.cudagraphs = args.cudagraphs
    cfg.device = args.agent_device or args.device
    cfg.buffer_device = args.buffer_device
    cfg.env_id = args.env_id
    cfg.env_cfg_entry_point = args.env_cfg_entry_point
    cfg.configure_env_entry_point = args.configure_env_entry_point
    cfg.expert_buffer_entry_point = args.expert_buffer_entry_point
    cfg.agent_class_entry_point = args.agent_class_entry_point
    cfg.agent_config_builder_entry_point = args.agent_config_builder_entry_point
    cfg.evaluate = args.evaluate
    cfg.eval_every_steps = args.eval_every_steps
    cfg.tracking_eval_entry_point = args.tracking_eval_entry_point
    cfg.tracking_eval_num_envs = args.tracking_eval_num_envs
    cfg.tracking_eval_motions = args.tracking_eval_motions
    cfg.tracking_eval_motions_root = args.tracking_eval_motions_root
    cfg.tracking_eval_max_motions = args.tracking_eval_max_motions
    cfg.tracking_eval_max_steps = args.tracking_eval_max_steps
    cfg.tracking_eval_mean_action = args.tracking_eval_mean_action
    cfg.tracking_eval_progress = args.tracking_eval_progress
    cfg.model = args.model
    cfg.hidden_dim = args.hidden_dim
    cfg.hidden_layers = args.hidden_layers
    cfg.z_dim = args.z_dim
    cfg.seq_length = args.seq_length
    cfg.actor_std = args.actor_std
    cfg.batch_size = args.batch_size
    cfg.discount = args.discount
    cfg.update_z_every_step = args.update_z_every_step
    cfg.lr_f = args.lr_f
    cfg.lr_b = args.lr_b
    cfg.lr_actor = args.lr_actor
    cfg.lr_critic = args.lr_critic
    cfg.lr_aux_critic = args.lr_aux_critic
    cfg.lr_discriminator = args.lr_discriminator
    cfg.weight_decay = args.weight_decay
    cfg.clip_grad_norm = args.clip_grad_norm
    cfg.fb_target_tau = args.fb_target_tau
    cfg.critic_target_tau = args.critic_target_tau
    cfg.ortho_coef = args.ortho_coef
    cfg.train_goal_ratio = args.train_goal_ratio
    cfg.fb_pessimism_penalty = args.fb_pessimism_penalty
    cfg.actor_pessimism_penalty = args.actor_pessimism_penalty
    cfg.critic_pessimism_penalty = args.critic_pessimism_penalty
    cfg.aux_critic_pessimism_penalty = args.aux_critic_pessimism_penalty
    cfg.stddev_clip = args.stddev_clip
    cfg.expert_asm_ratio = args.expert_asm_ratio
    cfg.relabel_ratio = args.relabel_ratio
    cfg.reg_coeff = args.reg_coeff
    cfg.reg_coeff_aux = args.reg_coeff_aux
    cfg.scale_reg = args.scale_reg
    cfg.q_loss_coef = args.q_loss_coef
    cfg.grad_penalty_discriminator = args.grad_penalty_discriminator
    cfg.weight_decay_discriminator = args.weight_decay_discriminator
    cfg.use_mix_rollout = args.use_mix_rollout
    cfg.z_buffer_size = args.z_buffer_size
    cfg.inference_batch_size = args.inference_batch_size
    cfg.norm_obs = args.norm_obs
    cfg.norm_z = args.norm_z
    cfg.b_norm = args.b_norm
    cfg.b_hidden_dim = args.b_hidden_dim
    cfg.b_hidden_layers = args.b_hidden_layers
    cfg.embedding_layers = args.embedding_layers
    cfg.num_parallel = args.num_parallel
    cfg.ensemble_mode = args.ensemble_mode
    cfg.discriminator_hidden_dim = args.discriminator_hidden_dim
    cfg.discriminator_hidden_layers = args.discriminator_hidden_layers
    cfg.norm_aux_reward_translate = args.norm_aux_reward_translate
    cfg.norm_aux_reward_scale = args.norm_aux_reward_scale
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
        output_name = "tmp_fbcpr_g1" if cfg.task == "g1" else "tmp_fbcpr"
        work_dir = Path.cwd() / "logs" / output_name / tmp_name
        cfg.work_dir = str(work_dir)
    else:
        work_dir = Path(cfg.work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)
    return work_dir


def make_tracking_evaluator(cfg: TrainConfig, agent, raw_env):
    if not cfg.evaluate or not cfg.tracking_eval_entry_point:
        return None
    factory = load_object(cfg.tracking_eval_entry_point)
    return factory(cfg=cfg, agent=agent, env=raw_env)


def make_env_and_expert_loader(cfg: TrainConfig, args_cli):
    import gymnasium as gym

    env_cfg = load_object(cfg.env_cfg_entry_point)()
    configure_env_cfg = load_object(cfg.configure_env_entry_point)
    make_expert_buffer = load_object(cfg.expert_buffer_entry_point)
    env_cfg = configure_env_cfg(env_cfg=env_cfg, cfg=cfg, args_cli=args_cli)
    raw_env = gym.make(cfg.env_id, cfg=env_cfg)
    expert_loader = lambda sequence_length: make_expert_buffer(
        cfg=cfg,
        raw_env=raw_env,
        sequence_length=sequence_length,
    )
    return raw_env, expert_loader


def main() -> None:
    args_cli, cfg = parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import wandb
    except ImportError:  # pragma: no cover
        wandb = None

    import bfm.tasks  # noqa: F401
    from agents.metamotivo.buffers.buffers import DictBuffer
    from agents.runner.env import IsaacLabVecEnvAdapter
    from agents.runner.env_runner import CheckpointHook, EnvRunner, EvalHook, FBcprAdapter, LogHook, ProgressHook

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
                name=f"fbcpr-{cfg.task}-{work_dir.name}",
                config=dataclasses.asdict(cfg),
                anonymous="allow",
            )
        except Exception as exc:
            print(f"[WARN] wandb init failed, continuing without online logging: {exc}", flush=True)
            cfg.use_wandb = False

    raw_env, expert_loader = make_env_and_expert_loader(cfg, args_cli)
    env = IsaacLabVecEnvAdapter(raw_env)
    obs, _ = env.reset()
    obs_dim = obs.shape[1]
    action_dim = env.num_actions

    build_fbcpr_agent_config = load_object(cfg.agent_config_builder_entry_point)
    agent_cfg = build_fbcpr_agent_config(obs_dim=obs_dim, action_dim=action_dim, cfg=cfg)
    agent_cls = load_object(cfg.agent_class_entry_point)
    agent = agent_cls(**dataclasses.asdict(agent_cfg))
    evaluator = make_tracking_evaluator(cfg, agent=agent, raw_env=raw_env)

    print("Loading expert trajectories", flush=True)
    expert_buffer = expert_loader(agent_cfg.model.seq_length)
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
        ProgressHook(enabled=cfg.progress, desc=f"{cfg.task} rollout"),
        EvalHook(every_steps=cfg.eval_every_steps, enabled=cfg.evaluate, use_wandb=cfg.use_wandb, evaluator=evaluator),
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
