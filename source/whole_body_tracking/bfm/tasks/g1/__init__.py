import gymnasium as gym

from . import g1_env_cfg


gym.register(
    id="G1-LAFAN-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g1_env_cfg.G1LafanEnvCfg,
    },
)
