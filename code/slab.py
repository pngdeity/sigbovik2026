#!/usr/bin/env python3
"""Minimum-mass axisymmetric slab with custom autograd kernel.

SlabField.apply(b) computes (g_z, g_r) with:
  forward:  8-point midpoint sum over z
  backward: analytic Jacobian via FTC (kernel evaluated once at z=b)

Loss function is arbitrary PyTorch; optimizer is standard Adam.
"""

import numpy as np
from datetime import datetime, UTC
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

torch.cuda.empty_cache()

d = {'device': 'cuda', 'dtype': torch.float32}
print(d)

# --- Parameters ---
disk_r    = 0.5
g0        = 1.0
epsilon   = 0.000625
n_src     = 8000
n_obs     = 800
n_z       = 25
smoothing = 5e-3
R_ext     = 3 * 4.0

# --- Elliptic integral lookup tables ---
N_table = 100_000
m_np  = np.linspace(0, 1 - 1e-7, N_table)
K_tbl = torch.tensor(ellipk(m_np), **d)
E_tbl = torch.tensor(ellipe(m_np), **d)

def elliptic_KE(m):
    m_c = torch.clamp(m, 0.0, 1.0 - 1e-7)
    idx = m_c * (N_table - 1)
    i0  = idx.long().clamp(0, N_table - 2)
    t   = idx - i0.float()
    K   = K_tbl[i0] * (1 - t) + K_tbl[i0 + 1] * t
    E   = E_tbl[i0] * (1 - t) + E_tbl[i0 + 1] * t
    return K, E

# --- Grids ---
r_src_np = np.linspace(0, R_ext, n_src + 1)
r_src_np = 0.5 * (r_src_np[:-1] + r_src_np[1:])
dr = float(R_ext / n_src)
r_obs_np = np.linspace(disk_r / n_obs, disk_r * 0.99, n_obs)

r_src = torch.tensor(r_src_np, **d)
r_obs = torch.tensor(r_obs_np, **d)

ro      = r_obs[:, None]
rs      = r_src[None, :]
ro_safe = ro.clamp(min=1e-10)

def _kernels(z):
    """Kz, Kr at positive depth z (broadcasts to (n_obs, n_src))."""
    alpha2 = (ro - rs)**2 + z**2
    beta2  = (ro + rs)**2 + z**2
    beta   = torch.sqrt(beta2.clamp(min=1e-30))
    k2     = (4 * ro * rs / beta2.clamp(min=1e-30)).clamp(0, 1 - 1e-7)
    K, E   = elliptic_KE(k2)
    Kz = z * 4 * E / (beta * alpha2.clamp(min=1e-20))
    Kr = 2 / (ro_safe * beta) * (
        (ro**2 - rs**2 - z**2) / alpha2.clamp(min=1e-20) * E + K)
    return Kz, Kr


class SlabField(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b):
        """g_z, g_r via midpoint sum over z (no graph stored)."""
        with torch.no_grad():
            bv = b[None, :]
            wt = bv * rs * dr / n_z
            gz = torch.zeros(n_obs, **d)
            gr = torch.zeros(n_obs, **d)
            for k in range(n_z):
                Kz, Kr = _kernels(bv * ((k + 0.5) / n_z))
                gz = gz + (Kz * wt).sum(1)
                gr = gr + (Kr * wt).sum(1)
        ctx.save_for_backward(b)
        return gz, gr

    @staticmethod
    def backward(ctx, grad_gz, grad_gr):
        """Analytic Jacobian-vector product: J^T @ grad via FTC."""
        (b,) = ctx.saved_tensors
        with torch.no_grad():
            Jz, Jr = _kernels(b[None, :])
            Jz = Jz * (rs * dr)   # (n_obs, n_src)
            Jr = Jr * (rs * dr)
            grad_b = grad_gz @ Jz + grad_gr @ Jr   # (n_src,)
        return grad_b


# --- Validation ---
with torch.no_grad():
    b_test = torch.full((n_src,), 0.1, **d)
    gz_t, gr_t = SlabField.apply(b_test)
    print(f"Validation — uniform slab b=0.1:")
    print(f"  ε = {epsilon:.4f}")
    print(f"  g_z = {gz_t.mean().item():.4f} (infinite-slab limit: {2*np.pi*0.1:.4f})")
    print(f"  g_r max = {gr_t.abs().max().item():.4f}")
    b0_val = float(abs(g0) / gz_t.mean().item() * 0.1)
    print(f"  Estimated b0 for |g_z|=1: {b0_val:.4f}")

# --- Optimization ---
log_b = torch.full((n_src,), np.log(b0_val), **d, requires_grad=True)

n_steps = 40000

optimizer = torch.optim.Adam([log_b], lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4)

for step in range(n_steps):
    optimizer.zero_grad()
    b = torch.exp(log_b)

    # geometric lambda ramp: 1e4 → 1e11 over training
    lam = 10 ** (4 + 7 * step / (n_steps - 1))

    gz, gr = SlabField.apply(b)

    mass    = 2 * np.pi * (b * r_src * dr).sum()
    err     = torch.sqrt((gz - g0)**2 + gr**2)
    penalty = torch.relu(err - epsilon).pow(2).mean()
    smooth  = smoothing * (log_b.diff() / dr).pow(2).mean() * R_ext
    loss    = mass + lam * penalty + smooth

    loss.backward()
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        log_b.clamp_(-8, 2)

    if step % 500 == 0:
        print(f"  step={step:4d}: mass={mass.item():.4f}, "
              f"max_err={err.max().item():.4f}, lam={lam:.0e}")

# --- Final evaluation ---
# %% final
with torch.no_grad():
    b_opt = torch.exp(log_b)
    gz_opt, gr_opt = SlabField.apply(b_opt)
    err_opt = torch.sqrt((gz_opt - g0)**2 + gr_opt**2)

    b_np   = b_opt.cpu().numpy()
    gz_np  = gz_opt.cpu().numpy()
    gr_np  = gr_opt.cpu().numpy()
    err_np = err_opt.cpu().numpy()
    mass_final = float(2 * np.pi * (b_opt * r_src * dr).sum().item())

print(f"\n--- Final ---")
print(f"Mass:               {mass_final:.4f}")
print(f"Max field error:    {err_np.max():.5f}  (ε={epsilon})")

# --- Leakage check ---
from solution_check import check_boundary_leakage
z_grid = np.linspace(0, b_np.max(), 200)
rho_2d = (z_grid[None, :] <= b_np[:, None]).astype(float)
leak = check_boundary_leakage(rho_2d, r_src_np, z_grid, tol=0.001)
print(f"Boundary fractions: { {k: f'{v:.4f}' for k, v in leak['boundary_fractions'].items()} }")
if leak['leaking']:
    raise RuntimeError(
        f"Mass leaking at boundary! fractions={leak['boundary_fractions']}"
    )
print("Leakage check: OK")

# --- Plot ---
# %% plot

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
r_plot = np.concatenate([-r_src_np[::-1], r_src_np])
b_plot = np.concatenate([b_np[::-1], b_np])
ax.fill_between(r_plot, 0, -b_plot, color='steelblue', alpha=0.8)
ax.plot(r_plot, -b_plot, color='black', linewidth=0.1)
ax.plot([-disk_r, disk_r], [0, 0], 'r-', lw=3, label='Disk')
ax.axvline( disk_r, color='r', ls='--', alpha=0.3)
ax.axvline(-disk_r, color='r', ls='--', alpha=0.3)
ax.set_xlim(-R_ext/2, R_ext/2)
ax.set_ylim(-b_np.max()*1.3, b_np.max()*0.3)
ax.set_aspect(0.25 * R_ext / (b_np.max()))
ax.set_title(f'(mass={mass_final:.4f}, ε={epsilon}, actual ε_max={err_np.max():.5f})')
ax.set_xlabel('r'); ax.set_ylabel('z'); ax.legend()

# ax = axes[1]
# ax.plot(r_obs_np, gz_np, 'b-', label='$g_z$')
# ax.plot(r_obs_np, gr_np, 'r-', label='$g_r$')
# ax.axhline(g0, color='b', ls='--', alpha=0.5)
# ax.axhline(0,  color='r', ls='--', alpha=0.5)
# ax.fill_between(r_obs_np, g0 - epsilon, g0 + epsilon, alpha=0.1, color='blue')
# ax.set_title('Field on disk'); ax.set_xlabel('r'); ax.legend()

ax = axes[1]
ax.plot(r_obs_np, err_np, 'k-')
ax.axhline(epsilon, color='r', ls='--', label=f'ε={epsilon}')
ax.set_title('|g - target|'); ax.set_xlabel('r'); ax.legend()
ax.set_ylim([0, 1.1 * epsilon])

plt.tight_layout()
fig.text(0.01, 0.01, text:=f'n_src={n_src}  n_obs={n_obs}  n_z={n_z} smooth={smoothing:.2e}',
         fontsize=8, color='gray', va='bottom', ha='left')
print(text)
plt.savefig('/www/flatearth/minmass.png', dpi=75)
plt.savefig(f'/www/flatearth/archive/minmass_{datetime.now(UTC).isoformat()}.png', dpi=150)
print("Saved to /www/flatearth/minmass.png")

# save settings and error for later reference
with open('/www/flatearth/minmass.tsv', 'a') as f:
    variables = [
        mass_final, epsilon, err_np.max(),
        disk_r, g0, epsilon, n_src, n_obs,
        n_z, smoothing, R_ext,
    ]
    f.write('\t'.join(map(str, variables)) + '\n')


for f in ['/www/flatearth_results.npz', f'/www/flatearth/archive_vars/{datetime.now(UTC).isoformat()}.npz']:
    np.savez(
        f,
        b_opt=b_np, r_src=r_src_np,
        gz=gz_np, gr=gr_np, err=err_np, r_obs=r_obs_np,
        epsilon=np.float64(epsilon),
        g0=np.float64(g0),
        disk_r=np.float64(disk_r),
        n_src=np.int32(n_src),
        n_obs=np.int32(n_obs),
        n_z=np.int32(n_z),
        smoothing=np.float64(smoothing),
        R_ext=np.float64(R_ext),
    )
print(f"Saved results to /www/flatearth_result.npz")

