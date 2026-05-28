from __future__ import annotations

from dataclasses import field

from isaaclab.utils import configclass


@configclass
class HumEnvFBcprRunnerCfg:
    seed: int = 0
    motions: str = ""
    motions_root: str = ""
    buffer_size: int = 5_000_000
    online_parallel_envs: int = 50
    log_every_updates: int = 5000
    work_dir: str | None = None
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


def build_fbcpr_agent_config(obs_dim: int, action_dim: int, cfg: HumEnvFBcprRunnerCfg):
    from agents.metamotivo.fb_cpr import FBcprAgentConfig

    agent_config = FBcprAgentConfig()
    agent_config.model.obs_dim = obs_dim
    agent_config.model.action_dim = action_dim
    agent_config.model.norm_obs = True
    agent_config.model.actor_std = cfg.actor_std
    agent_config.model.seq_length = cfg.seq_length
    agent_config.model.device = cfg.device

    agent_config.train.batch_size = cfg.batch_size
    agent_config.train.use_mix_rollout = 1
    agent_config.train.update_z_every_step = 150
    agent_config.train.discount = 0.98

    agent_config.model.archi.z_dim = cfg.z_dim
    agent_config.model.archi.b.norm = 1
    agent_config.model.archi.norm_z = 1
    agent_config.model.archi.f.hidden_dim = cfg.hidden_dim
    agent_config.model.archi.b.hidden_dim = 256
    agent_config.model.archi.actor.hidden_dim = cfg.hidden_dim
    agent_config.model.archi.critic.hidden_dim = cfg.hidden_dim
    agent_config.model.archi.f.hidden_layers = cfg.hidden_layers
    agent_config.model.archi.b.hidden_layers = 1
    agent_config.model.archi.actor.hidden_layers = cfg.hidden_layers
    agent_config.model.archi.critic.hidden_layers = cfg.hidden_layers
    agent_config.model.archi.f.model = cfg.model
    agent_config.model.archi.actor.model = cfg.model
    agent_config.model.archi.critic.model = cfg.model

    agent_config.train.lr_f = 1e-4
    agent_config.train.lr_b = 1e-5
    agent_config.train.lr_actor = 1e-4
    agent_config.train.lr_critic = 1e-4
    agent_config.train.ortho_coef = 100
    agent_config.train.train_goal_ratio = 0.2
    agent_config.train.expert_asm_ratio = 0.6
    agent_config.train.relabel_ratio = 0.8
    agent_config.train.reg_coeff = 0.01
    agent_config.train.q_loss_coef = 0.1

    agent_config.train.grad_penalty_discriminator = 10
    agent_config.train.weight_decay_discriminator = 0
    agent_config.train.lr_discriminator = 1e-5
    agent_config.model.archi.discriminator.hidden_layers = 3
    agent_config.model.archi.discriminator.hidden_dim = 1024

    agent_config.compile = cfg.compile
    agent_config.cudagraphs = cfg.cudagraphs
    return agent_config
