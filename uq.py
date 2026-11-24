import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from math import erf, sqrt

from solver import generate_dataset, solve_navier_stokes
from models import BaselineMLP, FNO2d, train_model


def _z_for_conf(p: float) -> float:
    lo, hi = 0.0, 5.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if erf(mid / sqrt(2.0)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ensemble_predict(models: list[torch.nn.Module], w0: torch.Tensor, nu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    preds = []
    with torch.no_grad():
        for m in models:
            m.eval()
            preds.append(m(w0, nu))
    stacked = torch.stack(preds, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0).clamp_min(1e-6)


def calibration_metrics(mean: np.ndarray, std: np.ndarray, target: np.ndarray) -> dict:
    err = target - mean
    mse = float(np.mean(err ** 2))
    rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(target))
    z_abs = np.abs(err) / std

    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    observed = np.array([float(np.mean(z_abs <= _z_for_conf(p))) for p in levels])
    ece = float(np.mean(np.abs(observed - levels)))
    cov95 = float(observed[-1])
    return {
        "mse": mse,
        "rel_l2": rel_l2,
        "cov95": cov95,
        "ece": ece,
        "mean_std": float(np.mean(std)),
        "levels": levels.tolist(),
        "observed": observed.tolist(),
    }


def plot_results(
    id_data: dict,
    ood_data: dict,
    id_mean: np.ndarray,
    id_std: np.ndarray,
    ood_mean: np.ndarray,
    ood_std: np.ndarray,
    id_metrics: dict,
    ood_metrics: dict,
    out_dir: str = "figures",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), constrained_layout=True)
    rows = [
        ("In-Distribution", id_data["w"][0, -1], id_mean[0, -1], id_std[0, -1]),
        ("Out-of-Distribution", ood_data["w"][0, -1], ood_mean[0, -1], ood_std[0, -1]),
    ]
    titles = ["Ground Truth w(x,y,T)", "FNO Ensemble Mean", "Abs Error |w - mu|", "Uncertainty 2*sigma"]
    for r, (label, truth, mean, std) in enumerate(rows):
        vmax = float(np.max(np.abs(truth)))
        err = np.abs(truth - mean)
        panels = [
            (truth, "RdBu_r", -vmax, vmax),
            (mean, "RdBu_r", -vmax, vmax),
            (err, "inferno", 0.0, float(np.max(err)) + 1e-6),
            (2.0 * std, "viridis", 0.0, float(np.max(2.0 * std)) + 1e-6),
        ]
        for c, (img, cmap, lo, hi) in enumerate(panels):
            im = axes[r, c].imshow(img.T, origin="lower", cmap=cmap, vmin=lo, vmax=hi, extent=[0, 2 * np.pi, 0, 2 * np.pi])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)
            if c == 0:
                axes[r, c].set_ylabel(label, fontsize=10, fontweight="bold")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(out_dir, "fields_and_uncertainty.png"), dpi=160)
    plt.close(fig)

    x = id_data["x"]
    mid = len(x) // 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)

    for label, truth, mean, std, color in [
        ("ID", id_data["w"][0, -1, :, mid], id_mean[0, -1, :, mid], id_std[0, -1, :, mid], "#1f77b4"),
        ("OOD", ood_data["w"][0, -1, :, mid], ood_mean[0, -1, :, mid], ood_std[0, -1, :, mid], "#d62728"),
    ]:
        ax1.plot(x, truth, "--", color=color, lw=1.6, label=f"{label} truth")
        ax1.plot(x, mean, "-", color=color, lw=1.6, label=f"{label} mean")
        ax1.fill_between(x, mean - 2 * std, mean + 2 * std, color=color, alpha=0.22, label=f"{label} +/-2s")
    ax1.set_title("Cross-Section Slice at y = pi (t = T)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Vorticity w")
    ax1.legend(ncol=2, fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot([0, 1], [0, 1], "k--", lw=1.2, label="Ideal calibration")
    ax2.plot(id_metrics["levels"], id_metrics["observed"], "o-", color="#1f77b4",
             label=f"ID (ECE={id_metrics['ece']:.3f}, avg std={id_metrics['mean_std']:.3f})")
    ax2.plot(ood_metrics["levels"], ood_metrics["observed"], "s-", color="#d62728",
             label=f"OOD (ECE={ood_metrics['ece']:.3f}, avg std={ood_metrics['mean_std']:.3f})")
    ax2.set_title("Uncertainty Calibration Curve")
    ax2.set_xlabel("Expected Coverage")
    ax2.set_ylabel("Observed Coverage")
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, "uq_calibration.png"), dpi=160)
    plt.close(fig)


def run_benchmark() -> dict:
    n, t_end, n_times = 32, 0.5, 6
    out_steps = n_times - 1
    dt = t_end / out_steps

    train = generate_dataset(240, n=n, t_end=t_end, n_times=n_times, nu_range=(5e-3, 2e-2), k_max=4, seed=0)
    test_id = generate_dataset(40, n=n, t_end=t_end, n_times=n_times, nu_range=(5e-3, 2e-2), k_max=4, seed=100)
    test_ood = generate_dataset(
        40, n=n, t_end=t_end, n_times=n_times, nu_range=(1e-3, 3e-3), k_max=6, forcing_amp=0.8, seed=200
    )

    t0 = time.perf_counter()
    solve_navier_stokes(test_id["w0"], nu=test_id["nu"], t_end=t_end, n_times=n_times)
    solver_ms = (time.perf_counter() - t0) * 1000.0 / len(test_id["w0"])

    w0_tr, nu_tr, y_tr = torch.from_numpy(train["w0"]), torch.from_numpy(train["nu"]), torch.from_numpy(train["w"][:, 1:])
    w0_id, nu_id, y_id = torch.from_numpy(test_id["w0"]), torch.from_numpy(test_id["nu"]), test_id["w"][:, 1:]
    w0_ood, nu_ood, y_ood = torch.from_numpy(test_ood["w0"]), torch.from_numpy(test_ood["nu"]), test_ood["w"][:, 1:]

    mlp = train_model(BaselineMLP(n, out_steps), w0_tr, nu_tr, y_tr, dt=dt, epochs=60, lr=2e-3, seed=0)
    mlp.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        mlp_id = mlp(w0_id, nu_id).numpy()
        mlp_ms = (time.perf_counter() - t0) * 1000.0 / len(w0_id)
        mlp_ood = mlp(w0_ood, nu_ood).numpy()

    ensemble = []
    for seed in range(5):
        torch.manual_seed(seed)
        m = FNO2d(out_steps=out_steps, width=24, modes=8, n_layers=3)
        ensemble.append(train_model(m, w0_tr, nu_tr, y_tr, dt=dt, epochs=50, lr=3e-3, pde_weight=1e-3, seed=seed))

    t0 = time.perf_counter()
    fno_id_mean, fno_id_std = ensemble_predict(ensemble, w0_id, nu_id)
    fno_ms = (time.perf_counter() - t0) * 1000.0 / len(w0_id)
    fno_ood_mean, fno_ood_std = ensemble_predict(ensemble, w0_ood, nu_ood)

    id_mean_np, id_std_np = fno_id_mean.numpy(), fno_id_std.numpy()
    ood_mean_np, ood_std_np = fno_ood_mean.numpy(), fno_ood_std.numpy()

    id_metrics = calibration_metrics(id_mean_np, id_std_np, y_id)
    ood_metrics = calibration_metrics(ood_mean_np, ood_std_np, y_ood)
    plot_results(test_id, test_ood, id_mean_np, id_std_np, ood_mean_np, ood_std_np, id_metrics, ood_metrics)

    results = {
        "solver_ms_per_traj": solver_ms,
        "mlp": {
            "ms_per_traj": mlp_ms,
            "id_mse": float(np.mean((mlp_id - y_id) ** 2)),
            "ood_mse": float(np.mean((mlp_ood - y_ood) ** 2)),
        },
        "fno_ensemble": {
            "ms_per_traj": fno_ms,
            "speedup_vs_solver": solver_ms / fno_ms,
            "id": id_metrics,
            "ood": ood_metrics,
        },
    }
    print(f"Solver:       {solver_ms:.2f} ms/traj")
    print(f"Baseline MLP: {mlp_ms:.3f} ms/traj | ID MSE={results['mlp']['id_mse']:.4e} | OOD MSE={results['mlp']['ood_mse']:.4e}")
    print(
        f"FNO Ensemble: {fno_ms:.3f} ms/traj ({results['fno_ensemble']['speedup_vs_solver']:.1f}x faster) | "
        f"ID MSE={id_metrics['mse']:.4e} (cov95={id_metrics['cov95']:.3f}, ECE={id_metrics['ece']:.3f}, std={id_metrics['mean_std']:.3f}) | "
        f"OOD MSE={ood_metrics['mse']:.4e} (cov95={ood_metrics['cov95']:.3f}, ECE={ood_metrics['ece']:.3f}, std={ood_metrics['mean_std']:.3f})"
    )
    return results


if __name__ == "__main__":
    run_benchmark()
