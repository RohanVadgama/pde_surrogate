import time
import numpy as np


def _wavenumbers(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kx = np.fft.fftfreq(n, d=1.0 / n)[:, None] # convert to spatial frequencies to make grad a simple multiplication by ik 
    ky = np.fft.rfftfreq(n, d=1.0 / n)[None, :]
    k2 = kx ** 2 + ky ** 2
    k2_safe = np.where(k2 == 0.0, 1.0, k2) #avoid divide by zero on laplacian
    
    # Multiplying two waves creates doubled frequencies too fast for the grid,
    # which fold back as fake low-frequency energy; zeroing the top 1/3 prevents this.
    dealias = (np.abs(kx) <= n / 3.0) & (np.abs(ky) <= n / 3.0)
    return kx, ky, k2, k2_safe, dealias


def _advection_hat(
    w_hat: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    k2_safe: np.ndarray,
    dealias: np.ndarray,
    n: int,
) -> np.ndarray:
    psi_hat = w_hat / k2_safe
    psi_hat[..., 0, 0] = 0.0
    u = np.fft.irfft2(1j * ky * psi_hat, s=(n, n), axes=(-2, -1))
    v = np.fft.irfft2(-1j * kx * psi_hat, s=(n, n), axes=(-2, -1))
    wx = np.fft.irfft2(1j * kx * w_hat, s=(n, n), axes=(-2, -1))
    wy = np.fft.irfft2(1j * ky * w_hat, s=(n, n), axes=(-2, -1))
    return -np.fft.rfft2(u * wx + v * wy, axes=(-2, -1)) * dealias


def solve_navier_stokes(
    w0: np.ndarray,
    nu: float | np.ndarray = 1e-2,
    t_end: float = 1.0,
    n_times: int = 11,
    dt: float = 5e-3,
    forcing: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    single = w0.ndim == 2
    w = np.asarray(w0, dtype=np.float64)
    if single:
        w = w[None, ...]
    batch, n, ny = w.shape
    if n != ny:
        raise ValueError("Grid must be square (n, n).")

    nu_arr = np.broadcast_to(np.asarray(nu, dtype=np.float64).reshape(-1, 1, 1), (batch, 1, 1))
    if np.any(nu_arr < 0):
        raise ValueError("Viscosity nu must be non-negative.")

    kx, ky, k2, k2_safe, dealias = _wavenumbers(n)
    w_hat = np.fft.rfft2(w, axes=(-2, -1))
    f_hat = 0.0 if forcing is None else np.fft.rfft2(np.asarray(forcing, dtype=np.float64), axes=(-2, -1)) * dealias

    t = np.linspace(0.0, t_end, n_times)
    traj = np.empty((batch, n_times, n, n), dtype=np.float64)
    traj[:, 0] = w

    for i in range(1, n_times):
        dt_span = t[i] - t[i - 1]
        n_steps = max(1, int(np.ceil(dt_span / dt)))
        h = dt_span / n_steps
        visc_num = 1.0 - 0.5 * h * nu_arr * k2
        visc_den = 1.0 + 0.5 * h * nu_arr * k2

        for _ in range(n_steps):
            n1 = _advection_hat(w_hat, kx, ky, k2_safe, dealias, n)
            w_star = (visc_num * w_hat + h * (n1 + f_hat)) / visc_den
            n2 = _advection_hat(w_star, kx, ky, k2_safe, dealias, n)
            w_hat = (visc_num * w_hat + 0.5 * h * (n1 + n2) + h * f_hat) / visc_den

        traj[:, i] = np.fft.irfft2(w_hat, s=(n, n), axes=(-2, -1))

    grid = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return (grid, t, traj[0]) if single else (grid, t, traj)


def sample_vorticity(
    n_samples: int,
    n: int = 32,
    k_max: int = 4,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(n, d=1.0 / n)[:, None]
    ky = np.fft.rfftfreq(n, d=1.0 / n)[None, :]
    k2 = kx ** 2 + ky ** 2
    mask = (k2 > 0) & (k2 <= k_max ** 2)
    scale = np.where(mask, (k2 + 1.0) ** -1.5, 0.0)

    real = rng.standard_normal((n_samples, n, n // 2 + 1))
    imag = rng.standard_normal((n_samples, n, n // 2 + 1))
    w_hat = (real + 1j * imag) * scale
    w0 = np.fft.irfft2(w_hat, s=(n, n), axes=(-2, -1))
    w0 /= np.std(w0, axis=(-2, -1), keepdims=True) + 1e-8
    return w0


def generate_dataset(
    n_samples: int,
    n: int = 32,
    t_end: float = 1.0,
    n_times: int = 11,
    nu_range: tuple[float, float] = (5e-3, 2e-2),
    k_max: int = 4,
    forcing_amp: float = 0.0,
    forcing_k: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    nu = np.exp(rng.uniform(np.log(nu_range[0]), np.log(nu_range[1]), size=n_samples))
    w0 = sample_vorticity(n_samples, n=n, k_max=k_max, seed=seed)

    forcing = None
    if forcing_amp != 0.0:
        grid = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        _, yy = np.meshgrid(grid, grid, indexing="ij")
        forcing = forcing_amp * np.sin(forcing_k * yy)

    grid, t, traj = solve_navier_stokes(w0, nu=nu, t_end=t_end, n_times=n_times, forcing=forcing)
    return {
        "x": grid.astype(np.float32),
        "t": t.astype(np.float32),
        "w0": w0.astype(np.float32),
        "nu": nu.astype(np.float32),
        "w": traj.astype(np.float32),
    }


def _verify_solver() -> None:
    n, nu, t_end = 32, 0.05, 1.0
    grid = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    w0_tg = 2.0 * np.cos(xx) * np.cos(yy)
    _, _, traj_tg = solve_navier_stokes(w0_tg, nu=nu, t_end=t_end, n_times=6, dt=5e-3)
    w_exact = w0_tg * np.exp(-2.0 * nu * t_end)
    tg_err = np.linalg.norm(traj_tg[-1] - w_exact) / np.linalg.norm(w_exact)
    assert tg_err < 1e-5, f"Taylor-Green error too large: {tg_err}"

    w0_inv = sample_vorticity(1, n=32, k_max=4, seed=7)[0]
    _, _, traj_inv = solve_navier_stokes(w0_inv, nu=0.0, t_end=0.5, n_times=6, dt=2e-3)
    z0 = 0.5 * np.mean(traj_inv[0] ** 2)
    z1 = 0.5 * np.mean(traj_inv[-1] ** 2)
    z_drift = abs(z1 - z0) / z0
    assert z_drift < 1e-3, f"Inviscid enstrophy drift too large: {z_drift}"

    print(f"Taylor-Green rel L2 error: {tg_err:.3e}")
    print(f"Inviscid enstrophy drift:  {z_drift:.3e}")


if __name__ == "__main__":
    _verify_solver()

    t0 = time.perf_counter()
    train = generate_dataset(n_samples=200, n=32, t_end=1.0, n_times=11, seed=42)
    elapsed = time.perf_counter() - t0
    print(f"Generated {train['w'].shape[0]} trajectories of shape {train['w'].shape[1:]} in {elapsed:.3f}s")
