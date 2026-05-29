from bfm.config.fb_cpr import G1FBcprRunnerCfg, build_fbcpr_agent_config, build_fbcpr_aux_agent_config
from bfm.tasks.g1.g1_env_cfg import DEFAULT_LAFAN_MOTIONS


def configure_env_cfg(env_cfg, cfg, args_cli):
    if not cfg.motions:
        cfg.motions = DEFAULT_LAFAN_MOTIONS
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.online_parallel_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.commands.motion.motions = cfg.motions
    env_cfg.commands.motion.reset_robot_state = cfg.motion_reset
    return env_cfg


def make_expert_buffer(cfg, raw_env, sequence_length: int):
    from bfm.tasks.g1.mdp.commands import load_expert_trajectories

    motion_command = raw_env.unwrapped.command_manager.get_term("motion")
    return load_expert_trajectories(
        cfg.motions,
        body_indexes=motion_command.body_indexes.detach().cpu().tolist(),
        device=cfg.buffer_device,
        sequence_length=sequence_length,
    )

__all__ = [
    "G1FBcprRunnerCfg",
    "build_fbcpr_agent_config",
    "build_fbcpr_aux_agent_config",
    "configure_env_cfg",
    "make_expert_buffer",
]
