#!/usr/bin/env python3
"""Minimum-mass axisymmetric slab, soft-constraint penalty method.

Find b(r) minimizing mass subject to:
  sqrt((g_z(r) - g0)^2 + g_r(r)^2) <= epsilon   for all r <= R

Uses PyTorch + Adam. K(k²) and E(k²) precomputed as lookup tables;
differentiable interpolation used at runtime so autograd works through them.

Soft penalty: L = mass + lambda * mean(relu(err - epsilon)^2)
Lambda is increased until no constraint violations remain.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from scipy.special import ellipk, ellipe

d = {'device': 'cpu', 'dtype': torch.float32}
print(d)

# --- Parameters ---
disk_r = 0.5
g0     = 1.0     # target field magnitude (kernel convention: positive downward)
epsilon = 0.050   # field vector tolerance

n_src = 500
n_obs = 500
n_z   = 600       # z quadrature points
R_ext = disk_r * 5.0

# --- Precompute elliptic integral lookup tables ---
N_table = 100_000
m_np  = np.linspace(0, 1 - 1e-7, N_table)
K_np  = ellipk(m_np)
E_np  = ellipe(m_np)

m_tbl = torch.tensor(m_np, **d)
K_tbl = torch.tensor(K_np, **d)
E_tbl = torch.tensor(E_np, **d)

def elliptic_KE(m):
    """Differentiable K(m) and E(m) via linear interpolation of precomputed table.

    Gradient flows through the interpolation weight, not the table index.
    Accurate to ~1e-6 with N_table=100k.
    """
    m_c = torch.clamp(m, 0.0, 1.0 - 1e-7)
    idx = m_c * (N_table - 1)
    i0  = idx.long().clamp(0, N_table - 2)
    i1  = (i0 + 1).clamp(0, N_table - 1)
    t   = idx - i0.float()          # interpolation weight (carries gradient)
    K   = K_tbl[i0] * (1 - t) + K_tbl[i1] * t
    E   = E_tbl[i0] * (1 - t) + E_tbl[i1] * t
    return K, E


# --- Grids (fixed, as plain tensors) ---
r_src_np = np.linspace(0, R_ext, n_src + 1)
r_src_np = 0.5 * (r_src_np[:-1] + r_src_np[1:])
dr = float(R_ext / n_src)

r_obs_np = np.linspace(disk_r / n_obs, disk_r * 0.99, n_obs)
z_frac   = ((np.arange(n_z) + 0.5) / n_z).astype(np.float32)

r_src  = torch.tensor(r_src_np, **d)
r_obs  = torch.tensor(r_obs_np, **d)
z_frac = torch.tensor(z_frac,   **d)


def compute_field(b_vals):
    """Compute g_z, g_r at r_obs for slab with bottom at b_vals.

    Args:
        b_vals: (n_src,) tensor, depth at each source point
    Returns:
        gz, gr: (n_obs,) tensors
    """
    # Broadcast to (n_obs, n_src, n_z)
    ro = r_obs[:, None, None]          # (n_obs, 1, 1)
    rs = r_src[None, :, None]          # (1, n_src, 1)
    zf = z_frac[None, None, :]         # (1, 1, n_z)

    z_  = b_vals[None, :, None] * zf  # (n_obs, n_src, n_z)  — actual depths
    dz_ = b_vals[None, :, None] / n_z

    beta2  = (ro + rs)**2 + z_**2
    alpha2 = (ro - rs)**2 + z_**2
    alpha2 = torch.clamp(alpha2, min=1e-20)
    beta   = torch.sqrt(torch.clamp(beta2, min=1e-30))

    k2 = torch.clamp(4 * ro * rs / torch.clamp(beta2, min=1e-30), 0.0, 1.0 - 1e-7)

    K, E = elliptic_KE(k2)

    # Vertical field kernel: z' * 4E / (beta * alpha²)
    gz_k = z_ * 4 * E / (beta * alpha2)

    # Radial field kernel: 2/(r*beta) * [(r²-r'²-z'²)/alpha² * E + K]
    ro_safe = torch.clamp(ro, min=1e-10)
    gr_k = 2 / (ro_safe * beta) * ((ro**2 - rs**2 - z_**2) / alpha2 * E + K)

    # Integrate: dr * dz * r_src weight
    weight = dz_ * rs * dr   # (n_obs, n_src, n_z)
    gz = (gz_k * weight).sum(dim=(1, 2))
    gr = (gr_k * weight).sum(dim=(1, 2))
    return gz, gr


# --- Validate kernels ---
with torch.no_grad():
    b_test = torch.full((n_src,), 0.1, **d)
    gz_t, gr_t = compute_field(b_test)
    print(f"Validation — uniform slab b=0.1:")
    print(f"  g_z = {gz_t.mean().item():.4f} (infinite-slab limit: {2*np.pi*0.1:.4f})")
    print(f"  g_r max = {gr_t.abs().max().item():.4f}")

    b0_val = float(abs(g0) / gz_t.mean().item() * 0.1)
    print(f"  Estimated b0 for |g_z|=1: {b0_val:.4f}")

# --- Optimization ---
log_b = torch.full((n_src,), np.log(b0_val),
                   **d, requires_grad=True)

optimizer = torch.optim.Adam([log_b], lr=3e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-4)

def run_opt(lam, n_steps=2000, log_every=500):
    """Optimize with fixed penalty weight lambda."""
    for step in range(n_steps):
        optimizer.zero_grad()
        b = torch.exp(log_b)
        gz, gr = compute_field(b)

        mass = 2 * np.pi * (b * r_src * dr).sum()
        err  = torch.sqrt((gz - g0)**2 + gr**2)
        penalty = torch.relu(err - epsilon).pow(2).mean()
        # Tiny smoothness term to suppress null-space notches (1e-4 << mass ~0.6)
        # smooth = 1e-4 * ((log_b[1:] - log_b[:-1])**2).mean()
        loss = mass + lam * penalty # + smooth

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            log_b.clamp_(-8, 2)

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                viol = (err > epsilon).sum().item()
            print(f"  λ={lam:.0f} step={step:4d}: mass={mass.item():.4f}, "
                  f"max_err={err.max().item():.4f}, violations={viol}/{n_obs}")

# Progressive lambda schedule
for lam in [1000, 10000, 100000]:
    print(f"\n--- Lambda = {lam} ---")
    run_opt(lam, n_steps=3000, log_every=1000)

# --- Final evaluation ---
with torch.no_grad():
    b_opt = torch.exp(log_b)
    gz_opt, gr_opt = compute_field(b_opt)
    err_opt = torch.sqrt((gz_opt - g0)**2 + gr_opt**2)

    b_np  = b_opt.cpu().numpy()
    gz_np = gz_opt.cpu().numpy()
    gr_np = gr_opt.cpu().numpy()
    err_np = err_opt.cpu().numpy()
    mass_final = float(2 * np.pi * (b_opt * r_src * dr).sum().item())

print(f"\n--- Final ---")
print(f"Mass:               {mass_final:.4f}")
print(f"Max field error:    {err_np.max():.4f}  (ε={epsilon})")
print(f"Violations:         {np.sum(err_np > epsilon)}/{n_obs}")
print(f"g_z range:          [{gz_np.min():.4f}, {gz_np.max():.4f}]")
print(f"g_r max:            {np.abs(gr_np).max():.4f}")

# --- Plot ---
# %% plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
r_plot = np.concatenate([-r_src_np[::-1], r_src_np])
b_plot = np.concatenate([b_np[::-1], b_np])
ax.fill_between(r_plot, 0, -b_plot, color='steelblue', alpha=0.8)
ax.plot([-disk_r, disk_r], [0, 0], 'r-', lw=3, label='Disk')
ax.axvline( disk_r, color='r', ls='--', alpha=0.3)
ax.axvline(-disk_r, color='r', ls='--', alpha=0.3)
ax.set_xlim(-R_ext/2, R_ext/2)
ax.set_ylim(-b_np.max()*1.3, b_np.max()*0.3)
ax.set_aspect('equal')
ax.set_title(f'Min-mass slab (ε={epsilon}, actual ε_max={err_np.max():.3f})')
ax.set_xlabel('r'); ax.set_ylabel('z'); ax.legend()

ax = axes[1]
ax.plot(r_obs_np, gz_np, 'b-', label='$g_z$')
ax.plot(r_obs_np, gr_np, 'r-', label='$g_r$')
ax.axhline(g0, color='b', ls='--', alpha=0.5)
ax.axhline(0,  color='r', ls='--', alpha=0.5)
ax.fill_between(r_obs_np, g0 - epsilon, g0 + epsilon, alpha=0.1, color='blue')
ax.set_title('Field on disk'); ax.set_xlabel('r'); ax.legend()

ax = axes[2]
ax.plot(r_obs_np, err_np, 'ko')
ax.axhline(epsilon, color='r', ls='--', label=f'ε={epsilon}')
ax.set_title('|g - target|'); ax.set_xlabel('r'); ax.legend()

plt.tight_layout()
fig.text(0.01, 0.01, f'n_src={n_src}  n_obs={n_obs}  n_z={n_z}',
         fontsize=8, color='gray', va='bottom', ha='left')
plt.savefig('/www/flatearth_minmass.png', dpi=150)
print("Saved to /www/flatearth_minmass.png")
