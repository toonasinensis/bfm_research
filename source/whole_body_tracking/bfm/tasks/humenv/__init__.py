import gymnasium as gym

from . import humenv_env_cfg


gym.register(
    id="HumEnv-MJCF-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": humenv_env_cfg.HumEnvMjcfEnvCfg,
    },
)
