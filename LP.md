# LP / Ring Solution

## Idea

Drop the $b(r)$ slab parameterization entirely. Instead, optimize over a 2D
axisymmetric density field $\rho(r, z) \geq 0$ on a discrete grid
$(r_j, z_k)$. The forward model is linear in $\rho$:

$$g_z(r_i) = \sum_{j,k} A^z_{ijk}\, \rho_{jk}, \qquad
  g_r(r_i) = \sum_{j,k} A^r_{ijk}\, \rho_{jk}$$

where $A^z_{ijk}$, $A^r_{ijk}$ are the elliptic-integral kernels evaluated at
$(r_i, r_j, z_k)$, weighted by $r_j \Delta r \Delta z$. The mass objective is
also linear: $M = 2\pi \sum_{jk} \rho_{jk}\, r_j \Delta r \Delta z$.

This is a **linear program**:

$$\min_{\rho \geq 0} \; c^T \rho \quad \text{subject to} \quad
  \|A\rho - g_{\rm target}\|_\infty \leq \varepsilon$$

(or equivalently with linear inequality constraints on each field component).

## Why the Solution Should Be Sparse (Rings)

By the fundamental theorem of linear programming, any optimal vertex solution
has at most $n_{\rm active}$ nonzero variables, where $n_{\rm active}$ is the
number of active (binding) constraints at the optimum. The actual number of
rings is determined by the LP solution itself and should be read off directly
from the number of nonzero $\rho_{jk}$ — not estimated from SVD.

## Known Issue: Ill-Posedness

The unconstrained LP ($\rho \geq 0$, no upper bound) is **mathematically
ill-posed**. The kernel has a singularity: $K_z \sim \pi/(r' \cdot z')$ as
$z' \to 0$ with $r' = r$. This means a thin ring at $(r' = r_{\rm obs}, z' \to 0)$
produces finite field contribution while its mass goes to zero. The continuous LP
infimum is zero; the discrete LP on a finite grid converges to a surface sheet at
$z' = z_{\min}$ as resolution increases. This is physically valid (surface mass
distributions are real) but removes all information about optimal depth and
geometry.

**Fixes**: either impose a minimum-depth constraint $z' \geq z_{\min}$,
restrict to $\rho \in [0,1]$ with minimum depth, or reformulate directly as a
surface distribution $\sigma(r')$.

## Properties

- **No slab assumption**: $\rho(r,z)$ is free to place mass anywhere in the
  half-space, not just in a connected slab beneath the surface.
- **Potentially unique**: unlike the $b(r)$ formulation, the LP optimum may be
  unique (or have a low-dimensional solution set), since the extreme-point
  structure of the LP polytope is rigid.
- **Scales with grid size**: the LP has $n_r \times n_z$ variables and
  $2 n_{\rm obs}$ constraints. For $n_r = n_z = 200$ this is 40,000 variables,
  easily handled by a standard LP solver (e.g. `scipy.optimize.linprog` with
  HiGHS, or CVXPY).

## Implementation Notes

- Build $A^z$ and $A^r$ using the same elliptic integral kernels as
  `flatearth_minmass.py`, evaluated at each $(r_j, z_k)$ with $z_k > 0$
  ($z_k$ is depth below surface; pass $z_k$ directly as the kernel argument).
- The $\|\cdot\|_\infty$ field constraint linearizes to $2 n_{\rm obs}$
  inequality constraints on $[g_z - g_0, g_r]$.
- Start with coarse grid ($n_r = n_z = 50$) to verify sparsity of solution,
  then refine.
- The LP solution provides a natural comparison point for the $b(r)$
  minimum-mass solution: same objective, strictly larger feasible set.
