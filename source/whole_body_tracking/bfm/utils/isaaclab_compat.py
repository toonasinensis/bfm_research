from __future__ import annotations

from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


if hasattr(sim_utils, "MjcfFileCfg"):
    MjcfFileCfg = sim_utils.MjcfFileCfg
else:
    from isaaclab.sim import converters
    from isaaclab.sim.utils import clone
    from isaaclab.sim.spawners.from_files import from_files
    from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg
    from pxr import Usd, UsdPhysics

    def _keep_articulation_root(prim: Usd.Prim, relative_path: str | None) -> None:
        if relative_path is None:
            return

        root_path = prim.GetPath().AppendPath(relative_path.lstrip("/"))
        stage = prim.GetStage()
        preferred_prim = stage.GetPrimAtPath(root_path)
        if not preferred_prim or not preferred_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return

        for child_prim in Usd.PrimRange(prim):
            if child_prim == preferred_prim:
                continue
            if child_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                child_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    def _discard_worldbody(prim: Usd.Prim) -> None:
        worldbody_path = prim.GetPath().AppendPath("worldBody")
        prim.GetStage().RemovePrim(worldbody_path)

    @clone
    def spawn_from_mjcf(
        prim_path: str,
        cfg: "MjcfFileCfg",
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
    ) -> Usd.Prim:
        mjcf_loader = converters.MjcfConverter(cfg)
        prim = from_files._spawn_from_usd_file(prim_path, mjcf_loader.usd_path, cfg, translation, orientation)
        if cfg.discard_worldbody:
            _discard_worldbody(prim)
        _keep_articulation_root(prim, cfg.articulation_root_prim_path)
        return prim

    @configclass
    class MjcfFileCfg(FileCfg, converters.MjcfConverterCfg):
        """Compatibility spawner for IsaacLab versions without sim.MjcfFileCfg."""

        func: Callable = spawn_from_mjcf
        articulation_root_prim_path: str | None = None
        discard_worldbody: bool = False
