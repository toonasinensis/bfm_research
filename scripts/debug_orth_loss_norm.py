from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F


def gram_stats(x: torch.Tensor) -> dict[str, float]:
    batch_size, z_dim = x.shape
    cov = x @ x.T
    off_diag = 1.0 - torch.eye(batch_size, device=x.device, dtype=x.dtype)
    off_diag_sum = off_diag.sum().clamp_min(1.0)
    return {
        "mean_norm": x.norm(dim=-1).mean().item(),
        "diag_mean": cov.diag().mean().item(),
        "orth_loss_diag": (-cov.diag().mean()).item(),
        "offdiag_abs_mean": (cov * off_diag).abs().sum().div(off_diag_sum).item(),
        "offdiag_sq_mean": (cov * off_diag).pow(2).sum().div(off_diag_sum).item(),
        "orth_loss_offdiag": (0.5 * (cov * off_diag).pow(2).sum().div(off_diag_sum)).item(),
    }


def print_stats(title: str, stats: dict[str, float]) -> None:
    print(f"\n[{title}]")
    for key, value in stats.items():
        print(f"{key:>20s}: {value:10.6f}")


def demo(batch_size: int, z_dim: int, seed: int, device: str) -> None:
    torch.manual_seed(seed)
    raw = 2 * torch.rand(batch_size, z_dim, device=device) - 1.0

    unit = F.normalize(raw, dim=-1)
    metamotivo = math.sqrt(z_dim) * F.normalize(raw, dim=-1)

    print(f"batch_size={batch_size} z_dim={z_dim} sqrt(z_dim)={math.sqrt(z_dim):.6f}")
    print("orth_loss_diag = -mean(diag(B @ B.T))")
    print("orth_loss_offdiag = 0.5 * mean_{i!=j}((B_i dot B_j)^2)")
    print("random normalized theory:")
    print(f"  unit norm:        diag ~= 1,       offdiag ~= 0.5 / z_dim = {0.5 / z_dim:.6f}")
    print(f"  sqrt(d) norm:     diag ~= z_dim,   offdiag ~= 0.5 * z_dim = {0.5 * z_dim:.6f}")

    print_stats("raw Gaussian, no explicit norm", gram_stats(raw))
    print_stats("unit normalize: B = normalize(raw)", gram_stats(unit))
    print_stats("MetaMotivo norm: B = sqrt(z_dim) * normalize(raw)", gram_stats(metamotivo))

    if batch_size <= z_dim:
        q, _ = torch.linalg.qr(torch.randn(z_dim, z_dim, device=device))
        orthogonal_rows = math.sqrt(z_dim) * q[:batch_size]
        print_stats("constructed orthogonal rows, only possible when batch_size <= z_dim", gram_stats(orthogonal_rows))
    else:
        welch = 0.5 * z_dim * (batch_size - z_dim) / (batch_size - 1)
        print("\n[lower-bound intuition]")
        print(
            f"batch_size > z_dim, so all rows cannot be mutually orthogonal. "
            f"Welch lower bound for orth_loss_offdiag is about {welch:.6f}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show why MetaMotivo sqrt(z_dim) normalization gives diag=z_dim.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    demo(batch_size=args.batch_size, z_dim=args.z_dim, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
