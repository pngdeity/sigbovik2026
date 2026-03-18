# Flat-Earth Inverse Problem — Agent Overview

## Objective

Find an axisymmetric mass distribution $\rho(r',z') \geq 0$ in the half-space
$z' \leq 0$ such that the gravitational field on a disk of radius $R$ at $z=0$
is uniform in magnitude and vertical. The general forward model is:

$$g_z(r) = \int_0^{r_{\max}} \int_{-\infty}^{0} K_z(r,r',z')\,\rho(r',z')\;r'\,dr'\,dz'$$

$$g_r(r) = \int_0^{r_{\max}} \int_{-\infty}^{0} K_r(r,r',z')\,\rho(r',z')\;r'\,dr'\,dz'$$

where the kernels, obtained by azimuthal integration of the 3D Green's
function, are:

$$K_z(r,r',z') = \frac{4z'\,E(k^2)}{\beta\,\alpha^2}, \qquad
  K_r(r,r',z') = \frac{2}{r\,\beta}\left[\frac{r^2-r'^2-z'^2}{\alpha^2}E(k^2)+K(k^2)\right]$$

with $\alpha^2=(r-r')^2+z'^2$, $\beta^2=(r+r')^2+z'^2$, $k^2=4rr'/\beta^2$,
and $K$, $E$ the complete elliptic integrals of the first and second kind.

The field constraint is:

$$\sqrt{(g_z(r)-g_0)^2+g_r(r)^2} \leq \varepsilon \quad \forall\,r\in[0,R]$$

## Parameterizations of $\rho$

The forward model is linear in $\rho$, so the choice of parameterization
affects both tractability and the character of the solution.

**Continuous nonneg measure** ($\rho \geq 0$, free weights): minimum-mass is
a linear program over measures. Convex, tractable via discrete LP on a grid.
**However, this formulation is ill-posed**: $K_z \sim \pi/(r'\,z')$ as $z'\to0$
with $r'=r$, so a near-surface ring at $r'=r_{\rm obs}$ produces finite field
with mass $\to 0$ as $z'\to 0$. The LP infimum is zero; the discrete LP
converges to a surface sheet at $z'=z_{\min}$ as resolution increases.
See `LP.md`.

**Binary** ($\rho \in \{0,1\}$, unit density): binary integer program
(NP-hard). Solution consists of filled 2D blobs in the $(r,z)$ cross-section
(solid tori in 3D). Disconnected regions allowed.

**Slab** ($\rho = \mathbf{1}_{-b(r') \leq z' \leq 0}$, unit density): special
case of binary where the occupied region is connected and touches the surface.
Reduces to finding a 1D boundary $b(r') \geq 0$. Nonconvex but tractable
with gradient-based methods. **Current approach.** Has a large null space
(~4989 modes out of n_src ≈ 5000): many b(r) profiles produce identical
fields to within ε. Fine-scale ripples and notches in b(r) under the disk
(r < R) are null-space oscillations that don't affect the field but do reduce
mass; they are numerical artifacts of high n_src and can be suppressed with a
smoothness penalty. The Jacobian of the field with respect to the boundary is obtained by evaluating the kernels at $z'=-b(r')$:

$$\frac{\partial g_z(r)}{\partial b(r')} = \frac{4\,b(r')\,E(k^2)}{\beta\,\alpha^2}\cdot r', \qquad
  \frac{\partial g_r(r)}{\partial b(r')} = \frac{2r'}{r\,\beta}\left[\frac{r^2-r'^2-b(r')^2}{\alpha^2}E(k^2)+K(k^2)\right]$$

with $\alpha^2=(r-r')^2+b(r')^2$, $\beta^2=(r+r')^2+b(r')^2$, $k^2=4rr'/\beta^2$.

## Why Perfect Uniformity is Impossible

Requiring $g_z = g_0$ and $g_r = 0$ everywhere on the disk forces the
gravitational potential to satisfy $\nabla\Phi = (0,0,-g_0)$ on a continuous
surface. By Laplace's equation and unique continuation, this extends
$\Phi = -g_0 z + C$ throughout the upper half-space. But this potential does
not decay at infinity — contradicting the requirement that any finite-mass
distribution produce $\Phi \sim M/r$ at large $r$. Therefore **exact
uniformity on a continuous disk requires infinite mass**; $\varepsilon > 0$
is a genuine physical parameter.

## Non-Uniqueness

The field constraint is far underdetermined: many $\rho$ distributions produce
identical fields to within $\varepsilon$. SVD analysis suggests a small number
of well-determined modes (exact count depends on discretization and threshold).
**The shape is determined by the optimality criterion, not the physics.**

| Criterion | Result |
|-----------|--------|
| Min mass, slab $b(r)$ | Flanged slab; mass concentrated at $r\approx R$ |
| Min second moment | Deep central bowl |
| Min perimeter (TV) | Slab + rectangular flanges |
| LP over $\rho(r',z')\geq 0$ | Ill-posed: collapses to surface sheet (see `LP.md`) |

**Current approach: minimum mass** — the only criterion that is naturally
window-independent (distant mass contributes little field per unit mass).

## Current Loss Function

$$L = 2\pi\int b(r')\,r'\,dr' + \lambda\cdot\frac{1}{n_{\rm obs}}
     \sum_i \mathrm{ReLU}(\mathrm{err}_i-\varepsilon)^2$$

Penalty weight $\lambda$ is ramped up in stages until all constraints are
satisfied. Implemented in PyTorch (CUDA) with Adam; $b(r')$ parameterized as
$\exp(\ell(r'))$ to enforce positivity. See `code/flatearth_minmass.py`.

Latest plot is saved to `/www/flatearth/minimass.png`

Results saved to `/www/flatearth_result.npz` (keys: `b_opt`, `r_src`, `gz`,
`gr`, `err`, `r_obs`, `meta=[epsilon, g0, disk_r, n_src, n_obs]`).

## Next Steps

- **LP / ring solution** (`LP.md`): drop the $b(r)$ assumption; solve LP
  over $\rho(r,z)\geq 0$.
  - **Known issue**: the unconstrained LP is ill-posed — the kernel singularity
    at $r'=r$, $z'\to 0$ allows the LP to satisfy all field constraints with a
    surface sheet of vanishing mass. Requires either a minimum-depth constraint,
    $\rho\in[0,1]$ with minimum depth, or a reformulation as a surface
    distribution $\sigma(r')$.
- **Pareto frontier**: trace $M^*(\varepsilon)$ to extract scaling exponent.
- **SVD / null-space analysis** (`SVD.md`): characterize field-constrained
  vs. mass-driven features of $b(r)$. Work in progress.

## Files

| File | Purpose |
|------|---------|
| `code/flatearth_minmass.py` | Main optimizer (slab approach) |
| `code/flatearth_svd_opt.py` | SVD analysis at $b_{\rm opt}$ |
| `code/flatearth_3d_svd.py` | SVD linearized around flat slab |
| `SVD.md` | SVD analysis notes |
| `LP.md` (repo root) | LP / ring solution design |
