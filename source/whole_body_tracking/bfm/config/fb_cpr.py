from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class HumEnvFBcprRunnerCfg:
    task: str = "humenv"
    runner_cfg_entry_point: str = "bfm.config.fb_cpr:HumEnvFBcprRunnerCfg"
    seed: int = 0
    motions: str = ""
    motions_root: str = ""
    buffer_size: int = 5_000_000
    online_parallel_envs: int = 50
    log_every_updates: int = 5000
    work_dir: str | None = None
    run_name: str | None = None
    num_env_steps: int = 30_000_000
    update_agent_every: int | None = None
    num_seed_steps: int | None = None
    num_agent_updates: int | None = None
    checkpoint_every_steps: int = 1_0_000
    prioritization: bool = False

    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = "fbcpr_humenv"
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
    tracking_eval_print_assignments: bool = False
    tracking_eval_print_assignment_limit: int = 64

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
    motion_reset: bool = True
    fall_prob: float = 0.2
    fall_height: float = 0.35

    reward_eval_tasks: list[str] = field(
        default_factory=lambda: [
            "move-ego-0-0",
            "jump-2",
            "move-ego-0-2",
            "move-ego-90-2",
            "move-ego-180-2",
            "rotate-x-5-0.8",
            "rotate-y-5-0.8",
            "rotate-z-5-0.8",
        ]
    )

    def __post_init__(self):
        if self.update_agent_every is None:
            self.update_agent_every = 10 * self.online_parallel_envs
        if self.num_seed_steps is None:
            self.num_seed_steps = 1000 * self.online_parallel_envs
        if self.num_agent_updates is None:
            self.num_agent_updates = self.online_parallel_envs
        if self.prioritization:
            raise NotImplementedError("FB-CPR prioritization is intentionally disabled in the first bfm_research port.")


@dataclass
class G1FBcprRunnerCfg(HumEnvFBcprRunnerCfg):
    task: str = "g1"
    runner_cfg_entry_point: str = "bfm.config.fb_cpr:G1FBcprRunnerCfg"
    motions: str = "/home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan"
    motions_root: str = ""
    wandb_pname: str | None = "fbcpr_g1"
    buffer_size: int = 2000_000
    online_parallel_envs: int = 1024
    log_every_updates: int = 10_24 * 50
    num_env_steps: int = 384_000_000
    update_agent_every: int | None = 1024
    num_seed_steps: int | None = 10_240
    num_agent_updates: int | None = 16
    checkpoint_every_steps: int = 1024 * 500
    buffer_device: str = "cuda"
    env_id: str = "G1-LAFAN-Flat-v0"
    env_cfg_entry_point: str = "bfm.tasks.g1.g1_env_cfg:G1LafanEnvCfg"
    configure_env_entry_point: str = "bfm.tasks.g1.config.fb_cpr:configure_env_cfg"
    expert_buffer_entry_point: str = "bfm.tasks.g1.config.fb_cpr:make_expert_buffer"
    agent_class_entry_point: str = "agents.metamotivo.fb_cpr_aux:FBcprAuxAgent"
    agent_config_builder_entry_point: str = "bfm.config.fb_cpr:build_fbcpr_aux_agent_config"
    eval_every_steps: int = 1024 * 500
    tracking_eval_entry_point: str = "bfm.tasks.g1.eval.fbcpr_tracking:make_tracking_evaluator"
    tracking_eval_num_envs: int = 1024
    tracking_eval_motions: str = ""
    tracking_eval_motions_root: str = ""
    tracking_eval_max_motions: int = 80000
    tracking_eval_max_steps: int = 12800000
    tracking_eval_mean_action: bool = True
    tracking_eval_progress: bool = True

    model: str = "residual"
    hidden_dim: int = 2048
    hidden_layers: int = 6
    actor_std: float = 0.05
    update_z_every_step: int = 100
    lr_f: float = 3e-4
    lr_b: float = 1e-5
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_aux_critic: float = 3e-4
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
    reg_coeff: float = 0.05
    reg_coeff_aux: float = 0.02
    scale_reg: bool = True
    q_loss_coef: float = 0.0
    grad_penalty_discriminator: float = 10.0
    weight_decay_discriminator: float = 0.0
    use_mix_rollout: bool = True
    z_buffer_size: int = 8192
    aux_rewards: list[str] = field(
        default_factory=lambda: [
            "penalty_torques",
            "penalty_action_rate",
            "limits_dof_pos",
            "limits_torque",
            "penalty_undesired_contact",
            "penalty_feet_ori",
            "penalty_ankle_roll",
            "penalty_slippage",
        ]
    )
    aux_rewards_scaling: dict[str, float] = field(
        default_factory=lambda: {
            "penalty_action_rate": -0.1,
            "penalty_feet_ori": -0.4,
            "penalty_ankle_roll": -4.0,
            "limits_dof_pos": -10.0,
            "penalty_slippage": -2.0,
            "penalty_undesired_contact": -1.0,
            "penalty_torques": 0.0,
            "limits_torque": 0.0,
        }
    )


def _populate_fbcpr_agent_config(agent_config, obs_dim: int, action_dim: int, cfg: HumEnvFBcprRunnerCfg):
    agent_config.model.obs_dim = obs_dim
    agent_config.model.action_dim = action_dim
    agent_config.model.actor_std = cfg.actor_std
    agent_config.model.seq_length = cfg.seq_length
    agent_config.model.inference_batch_size = cfg.inference_batch_size
    agent_config.model.device = cfg.device
    agent_config.model.norm_obs = cfg.norm_obs

    agent_config.train.batch_size = cfg.batch_size
    agent_config.train.use_mix_rollout = cfg.use_mix_rollout
    agent_config.train.update_z_every_step = cfg.update_z_every_step
    agent_config.train.discount = cfg.discount
    agent_config.train.z_buffer_size = cfg.z_buffer_size

    agent_config.model.archi.z_dim = cfg.z_dim
    agent_config.model.archi.b.norm = cfg.b_norm
    agent_config.model.archi.norm_z = cfg.norm_z
    agent_config.model.archi.f.hidden_dim = cfg.hidden_dim
    agent_config.model.archi.b.hidden_dim = cfg.b_hidden_dim
    agent_config.model.archi.actor.hidden_dim = cfg.hidden_dim
    agent_config.model.archi.critic.hidden_dim = cfg.hidden_dim
    agent_config.model.archi.f.hidden_layers = cfg.hidden_layers
    agent_config.model.archi.b.hidden_layers = cfg.b_hidden_layers
    agent_config.model.archi.actor.hidden_layers = cfg.hidden_layers
    agent_config.model.archi.critic.hidden_layers = cfg.hidden_layers
    agent_config.model.archi.f.embedding_layers = cfg.embedding_layers
    agent_config.model.archi.actor.embedding_layers = cfg.embedding_layers
    agent_config.model.archi.critic.embedding_layers = cfg.embedding_layers
    agent_config.model.archi.f.num_parallel = cfg.num_parallel
    agent_config.model.archi.critic.num_parallel = cfg.num_parallel
    agent_config.model.archi.f.ensemble_mode = cfg.ensemble_mode
    agent_config.model.archi.critic.ensemble_mode = cfg.ensemble_mode
    agent_config.model.archi.f.model = cfg.model
    agent_config.model.archi.actor.model = cfg.model
    agent_config.model.archi.critic.model = cfg.model

    agent_config.train.lr_f = cfg.lr_f
    agent_config.train.lr_b = cfg.lr_b
    agent_config.train.lr_actor = cfg.lr_actor
    agent_config.train.lr_critic = cfg.lr_critic
    agent_config.train.weight_decay = cfg.weight_decay
    agent_config.train.clip_grad_norm = cfg.clip_grad_norm
    agent_config.train.fb_target_tau = cfg.fb_target_tau
    agent_config.train.critic_target_tau = cfg.critic_target_tau
    agent_config.train.ortho_coef = cfg.ortho_coef
    agent_config.train.train_goal_ratio = cfg.train_goal_ratio
    agent_config.train.fb_pessimism_penalty = cfg.fb_pessimism_penalty
    agent_config.train.actor_pessimism_penalty = cfg.actor_pessimism_penalty
    agent_config.train.critic_pessimism_penalty = cfg.critic_pessimism_penalty
    agent_config.train.stddev_clip = cfg.stddev_clip
    agent_config.train.expert_asm_ratio = cfg.expert_asm_ratio
    agent_config.train.relabel_ratio = cfg.relabel_ratio
    agent_config.train.reg_coeff = cfg.reg_coeff
    agent_config.train.scale_reg = cfg.scale_reg
    agent_config.train.q_loss_coef = cfg.q_loss_coef
    agent_config.train.grad_penalty_discriminator = cfg.grad_penalty_discriminator
    agent_config.train.weight_decay_discriminator = cfg.weight_decay_discriminator
    agent_config.train.lr_discriminator = cfg.lr_discriminator

    agent_config.model.archi.discriminator.hidden_layers = cfg.discriminator_hidden_layers
    agent_config.model.archi.discriminator.hidden_dim = cfg.discriminator_hidden_dim

    agent_config.compile = cfg.compile
    agent_config.cudagraphs = cfg.cudagraphs
    return agent_config


def build_fbcpr_agent_config(obs_dim: int, action_dim: int, cfg: HumEnvFBcprRunnerCfg):
    from agents.metamotivo.fb_cpr import FBcprAgentConfig

    return _populate_fbcpr_agent_config(FBcprAgentConfig(), obs_dim, action_dim, cfg)


def build_fbcpr_aux_agent_config(obs_dim: int, action_dim: int, cfg: HumEnvFBcprRunnerCfg):
    from agents.metamotivo.fb_cpr_aux import FBcprAuxAgentConfig

    agent_config = _populate_fbcpr_agent_config(FBcprAuxAgentConfig(), obs_dim, action_dim, cfg)
    agent_config.model.archi.aux_critic = copy.deepcopy(agent_config.model.archi.critic)
    agent_config.model.norm_aux_reward.translate = cfg.norm_aux_reward_translate
    agent_config.model.norm_aux_reward.scale = cfg.norm_aux_reward_scale
    agent_config.train.lr_aux_critic = cfg.lr_aux_critic
    agent_config.train.reg_coeff_aux = cfg.reg_coeff_aux
    agent_config.train.aux_critic_pessimism_penalty = cfg.aux_critic_pessimism_penalty
    agent_config.aux_rewards = list(cfg.aux_rewards)
    agent_config.aux_rewards_scaling = dict(cfg.aux_rewards_scaling)
    missing_scalings = sorted(set(agent_config.aux_rewards) - set(agent_config.aux_rewards_scaling))
    if missing_scalings:
        raise KeyError(f"Missing aux reward scaling for {missing_scalings}.")
    return agent_config


__all__ = [
    "HumEnvFBcprRunnerCfg",
    "G1FBcprRunnerCfg",
    "build_fbcpr_agent_config",
    "build_fbcpr_aux_agent_config",
]
