from __future__ import annotations

import collections
import dataclasses
import numbers
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _ensure_humenv_importable() -> None:
    try:
        import humenv  # noqa: F401
    except ImportError:
        local_humenv = Path("/home/chn/hajimi/humenv")
        if local_humenv.is_dir():
            sys.path.insert(0, str(local_humenv))
        try:
            import humenv  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MetaMotivo HumEnv bench eval requires the `humenv` package. "
                "Install it or add /home/chn/hajimi/humenv to PYTHONPATH before using "
                "bfm.tasks.humenv.eval.metamotivo_tracking:make_tracking_evaluator."
            ) from exc


@dataclasses.dataclass
class MetaMotivoHumEnvTrackingEvaluator:
    cfg: Any
    agent: Any

    def __call__(self, runner) -> dict[str, float]:
        del runner
        _ensure_humenv_importable()
        from agents.metamotivo.wrappers.humenvbench import TrackingWrapper
        from humenv.bench import TrackingEvaluation

        model = self.agent._model
        previous_training = model.training
        previous_device = torch.device(model.cfg.device)
        motions = self.cfg.tracking_eval_motions or self.cfg.motions
        motions_root = self.cfg.tracking_eval_motions_root or self.cfg.motions_root
        num_envs = max(int(self.cfg.tracking_eval_num_envs), 1)
        start_time = time.time()

        print(
            "[MetaMotivoHumEnvTrackingEvaluator] "
            f"motions={motions} num_envs={num_envs} mean_action=True",
            flush=True,
        )
        try:
            model.to("cpu")
            model.train(False)
            eval_agent = TrackingWrapper(model=model)
            tracking_eval = TrackingEvaluation(
                motions=motions,
                motion_base_path=motions_root,
                env_kwargs={"state_init": "Default"},
                num_envs=num_envs,
            )
            try:
                tracking_metrics = tracking_eval.run(agent=eval_agent)
            finally:
                close = getattr(tracking_eval, "close", None)
                if callable(close):
                    close()
        finally:
            model.to(previous_device)
            model.train(previous_training)

        aggregate: dict[str, list[float]] = collections.defaultdict(list)
        for metric in tracking_metrics.values():
            for key, value in metric.items():
                if isinstance(value, numbers.Number):
                    aggregate[key].append(float(value))
        output = {f"tracking/{key}": float(np.mean(value)) for key, value in sorted(aggregate.items())}
        output.update(
            {f"tracking/{key}#std": float(np.std(value)) for key, value in sorted(aggregate.items())}
        )
        output["tracking/num_motions"] = float(len(tracking_metrics))
        output["tracking/time"] = time.time() - start_time
        print(f"[MetaMotivoHumEnvTrackingEvaluator] done metrics={output}", flush=True)
        return output


def make_tracking_evaluator(cfg, agent, env=None):
    del env
    return MetaMotivoHumEnvTrackingEvaluator(cfg=cfg, agent=agent)
