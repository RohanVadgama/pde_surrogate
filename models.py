import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2
        )
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNO2d(nn.Module):
    def __init__(self, out_steps: int, width: int = 24, modes: int = 8, n_layers: int = 3):
        super().__init__()
        self.lift = nn.Conv2d(4, width, kernel_size=1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_steps, kernel_size=1),
        )

    def forward(self, w0: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        b, h, w = w0.shape
        nu_feat = torch.log10(nu).view(b, 1, 1, 1).expand(b, 1, h, w) + 2.0
        gx = torch.linspace(0.0, 1.0, h, device=w0.device).view(1, 1, h, 1).expand(b, 1, h, w)
        gy = torch.linspace(0.0, 1.0, w, device=w0.device).view(1, 1, 1, w).expand(b, 1, h, w)
        x = torch.cat([w0.unsqueeze(1), nu_feat, gx, gy], dim=1)

        x = self.lift(x)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = F.gelu(spec(x) + pw(x))
        return self.proj(x)


class BaselineMLP(nn.Module):
    def __init__(self, n: int, out_steps: int, hidden: int = 256):
        super().__init__()
        self.n = n
        self.out_steps = out_steps
        self.net = nn.Sequential(
            nn.Linear(n * n + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_steps * n * n),
        )

    def forward(self, w0: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        b = w0.shape[0]
        nu_feat = torch.log10(nu).view(b, 1) + 2.0
        x = torch.cat([w0.reshape(b, -1), nu_feat], dim=1)
        return self.net(x).view(b, self.out_steps, self.n, self.n)


def navier_stokes_residual(w0: torch.Tensor, w_pred: torch.Tensor, nu: torch.Tensor, dt: float) -> torch.Tensor:
    w = torch.cat([w0.unsqueeze(1), w_pred], dim=1)
    n = w.shape[-1]
    dw_dt = (w[:, 1:] - w[:, :-1]) / dt
    w_mid = 0.5 * (w[:, 1:] + w[:, :-1])

    kx = torch.fft.fftfreq(n, d=1.0 / n, device=w.device).view(1, 1, n, 1)
    ky = torch.fft.rfftfreq(n, d=1.0 / n, device=w.device).view(1, 1, 1, n // 2 + 1)
    k2 = kx ** 2 + ky ** 2
    k2_safe = torch.where(k2 == 0.0, torch.ones_like(k2), k2)

    w_hat = torch.fft.rfft2(w_mid)
    psi_hat = w_hat / k2_safe
    psi_hat[..., 0, 0] = 0.0

    u = torch.fft.irfft2(1j * ky * psi_hat, s=(n, n))
    v = torch.fft.irfft2(-1j * kx * psi_hat, s=(n, n))
    wx = torch.fft.irfft2(1j * kx * w_hat, s=(n, n))
    wy = torch.fft.irfft2(1j * ky * w_hat, s=(n, n))
    lap_w = torch.fft.irfft2(-k2 * w_hat, s=(n, n))

    res = dw_dt + u * wx + v * wy - nu.view(-1, 1, 1, 1) * lap_w
    return (res ** 2).mean()


def train_model(
    model: nn.Module,
    w0: torch.Tensor,
    nu: torch.Tensor,
    w_target: torch.Tensor,
    dt: float,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-3,
    pde_weight: float = 0.0,
    seed: int = 0,
) -> nn.Module:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_samples = w0.shape[0]

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            pred = model(w0[idx], nu[idx])
            loss = F.mse_loss(pred, w_target[idx])
            if pde_weight > 0.0:
                loss = loss + pde_weight * navier_stokes_residual(w0[idx], pred, nu[idx], dt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model
