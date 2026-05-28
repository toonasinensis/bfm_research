from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two FB-CPR tracking rollout npz files.")
    parser.add_argument("--a", type=str, required=True, help="First rollout npz.")
    parser.add_argument("--b", type=str, required=True, help="Second rollout npz.")
    parser.add_argument("--label-a", type=str, default="a")
    parser.add_argument("--label-b", type=str, default="b")
    return parser.parse_args()


def _stats(value: np.ndarray) -> dict[str, float]:
    flat = value.astype(np.float64).reshape(value.shape[0], -1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2_per_step": float(np.linalg.norm(flat, axis=1).mean()),
    }


def _print_stats(label: str, data: np.lib.npyio.NpzFile) -> None:
    print(f"[{label}] shapes: " + ", ".join(f"{key}={data[key].shape}" for key in data.files))
    for key in ("obs", "action", "z"):
        if key not in data:
            continue
        stats = _stats(data[key])
        print(
            f"[{label}] {key}: "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
            f"min={stats['min']:.6f} max={stats['max']:.6f} "
            f"l2/step={stats['l2_per_step']:.6f}"
        )
        if key == "action":
            action = data[key]
            print(f"[{label}] action_saturation: {float((np.abs(action) >= 0.999).mean()):.6f}")
    if "obs" in data and "reference_observation" in data:
        obs = data["obs"][:, 0, :] if data["obs"].ndim == 3 else data["obs"]
        ref = data["reference_observation"][: obs.shape[0]]
        print(f"[{label}] obs_reference_mse: {float(np.mean((obs - ref) ** 2)):.6f}")
    if "done" in data:
        print(f"[{label}] done_sum: {int(data['done'].sum())}")


def _print_diff(label_a: str, a: np.lib.npyio.NpzFile, label_b: str, b: np.lib.npyio.NpzFile) -> None:
    for key in ("obs", "action", "z"):
        if key not in a or key not in b:
            continue
        if a[key].shape != b[key].shape:
            print(f"[diff] {key}: shape mismatch {a[key].shape} vs {b[key].shape}")
            continue
        delta = a[key] - b[key]
        print(
            f"[diff {label_a}-{label_b}] {key}: "
            f"mse={float(np.mean(delta ** 2)):.6f} "
            f"mae={float(np.mean(np.abs(delta))):.6f} "
            f"max_abs={float(np.max(np.abs(delta))):.6f}"
        )


def main() -> None:
    args = parse_args()
    path_a = Path(args.a)
    path_b = Path(args.b)
    data_a = np.load(path_a)
    data_b = np.load(path_b)
    _print_stats(args.label_a, data_a)
    _print_stats(args.label_b, data_b)
    _print_diff(args.label_a, data_a, args.label_b, data_b)


if __name__ == "__main__":
    main()
