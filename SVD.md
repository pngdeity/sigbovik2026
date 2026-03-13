# SVD Analysis of the Flat-Earth Inverse Problem

## Problem Statement

Given a finite disk of radius $R$ on a surface ($z=0$), find a mass distribution below the surface that produces a uniform vertical gravitational field $g_z = g_0$ on the disk. We restrict to axisymmetric solutions parameterized by a single boundary function $b(r)$: the mass occupies the volume of revolution between $z=0$ and $z=-b(r)$, with unit density.

The forward model is:

$$g_z(r) = \int_0^{r_{\max}} \left[\Psi(r,r',0) - \Psi(r,r',b(r'))\right] r' \, dr'$$

where $\Psi(r,r',d) = 4K(k)/\sqrt{(r+r')^2+d^2}$ is the azimuthally-integrated 3D Green's function and $K$ is the complete elliptic integral of the first kind.

### Why eps=0 works in 3D

In the 2D formulation (`gemini_2d_analytic.py`), the mass slab has a top boundary at $z=-\varepsilon$ with $\varepsilon>0$ required because the 2D line-mass kernel $1/\sqrt{(x-x')^2}$ diverges when source and observer coincide. This $\varepsilon$ is an arbitrary parameter — a regularization artifact, not physics.

In 3D, the kernel $\Psi(r,r',0)$ has $K(k)\to\infty$ when $r'=r$ and $d=0$, but this is a logarithmic singularity: $K \sim -\frac{1}{2}\ln|r-r'|$. Integrated against the smooth factor $r'\,dr'$, it is integrable. The vertical field from a thin surface slab is finite (this is the standard result $g_z = 2\pi G\rho\delta$ for an infinite slab of thickness $\delta$, which is finite for any $\delta > 0$ and goes to zero as $\delta\to 0$). So we can set $\varepsilon=0$ and eliminate one arbitrary parameter.

## The Non-Uniqueness Problem

### Previous approaches and what they revealed

We first solved this problem using pixel-based topology optimization in PyTorch (`gemini_2d_topo.py`, `gemini_2d_minmass.py`, `gemini_2d_minperim.py`). These codes optimize a 2D density field $m(x,y) \in [0,1]$ on a grid, using different regularizers:

| Regularizer | Code | Result |
|---|---|---|
| Compactness ($\min \sum m_i r_i^2$) | `gemini_2d_topo.py` | Concave bowl |
| Min perimeter ($\min \mathrm{TV}(m)$) | `gemini_2d_minperim.py` | Slab + rounded flanges |
| Min mass ($\min \sum m_i$) | `gemini_2d_minmass.py` | Thin slab + small flanges |

All three satisfy the field constraint equally well, but produce visibly different shapes. This established the central fact: **the shape is determined by the regularizer, not the physics.**

We then reduced the problem to finding a 1D boundary curve $b(r)$ (the bottom of the slab), which is more efficient and eliminates the 2D grid resolution as a confounding factor. We tried:

1. **Chebyshev expansion** of $b(r)$ with `least_squares` and a mass penalty. The Chebyshev basis became ill-conditioned: coefficients grew to $O(10^3)$, causing the boundary to diverge between quadrature nodes while appearing fine at the nodes themselves.

2. **Cubic spline on control nodes** with `least_squares`. Better conditioned, achieved 0.3–0.5% field uniformity. But when we then tried to *minimize mass* (the physically motivated objective), the optimizer carved notches in $b(r)$ — removing mass in regions where the smoothing integral couldn't detect the change. The minimum-mass boundary is pathological (fractal-like thinning toward measure zero).

3. **Smoothness penalty** ($\min \|b''\|^2$ subject to field constraint). Produces clean shapes, but the penalty weight is arbitrary — a different weight gives a different shape.

The common thread: every approach requires an arbitrary choice (regularizer type, penalty weight, number of basis functions) that determines the fine-scale shape. The question becomes: **which features of $b(r)$ are actually constrained by the field data, and which are artifacts of these choices?**

## The SVD Approach

### Linearization

Linearize the forward operator around a reference uniform slab of depth $b_0$:

$$\delta g_z(r_i) \approx \sum_j A_{ij} \, \delta b(r_j)$$

where

$$A_{ij} = -\frac{\partial \Psi}{\partial d}\bigg|_{d=b_0}(r_i, r_j) \cdot r_j \, \Delta r$$

This is valid when $|\delta b| \ll b_0$. The matrix $A$ is $(n_{\rm obs} \times n_{\rm src})$, typically $(39 \times 200)$.

### What the SVD reveals

The SVD $A = U \Sigma V^T$ decomposes the forward operator into independent modes:

- **Columns of $V$** (right singular vectors): modes in *source space* — independent shapes that $b(r)$ can take
- **Columns of $U$** (left singular vectors): corresponding modes in *data space* — the field patterns they produce
- **Singular values $\sigma_i$**: how strongly each source mode affects the field

The spectrum drops steeply: $\sigma_0 = 4.4$, falling below $10^{-2}$ by mode 6, and reaching $10^{-12}$ by mode 39. The condition number is $\sim 10^{12}$. **Only 6 modes have singular values above 1% of the maximum.**

### Interpretation

The 6 well-determined modes are smooth, low-frequency shapes concentrated near and within the disk ($r \lesssim R$). They represent the components of $b(r)$ that the surface field actually constrains — the "resolvable" features. These are the aspects of the boundary shape that are physically meaningful.

The remaining ~194 modes (poorly determined, $\sigma < 0.01 \sigma_{\max}$) are high-frequency oscillations, increasingly concentrated near the disk edge. They represent perturbations to $b(r)$ that are *invisible* to the surface field — the null space. Any combination of these can be added to a solution without affecting the field. This is why:

- The minimum-mass optimizer carves notches (adds null-space modes that remove mass)
- Different regularizers give different shapes (they project differently onto the null space)
- Increasing resolution reveals more artifacts (finer null-space modes become available)

### The truncated SVD solution

Using only the 6 well-determined modes gives the minimum-norm $\delta b$ that satisfies the field constraint in the resolved subspace:

$$\delta b_{\rm TSVD}(r) = \sum_{i=0}^{5} \frac{\mathbf{u}_i^T \, \delta\mathbf{g}}{\sigma_i} \, \mathbf{v}_i(r)$$

This is the unique solution with no null-space contamination. It achieves ~3% field uniformity (limited by linearization error and the truncation). Nonlinear refinement from this starting point can improve the fit while inheriting the smooth structure.

## Significance

1. **The number 6 is the answer.** The surface field on a disk of radius $R$ constrains exactly ~6 independent features of the subsurface boundary shape. Everything else — the notches, the fine-scale bowl structure, the exact flange profile — is unconstrained by the physics and determined entirely by the choice of regularizer.

2. **The well-determined modes are physically interpretable.** Mode 0 is roughly constant (overall depth). Mode 1 is the linear correction (deeper at center vs edges — the "bowl"). Higher modes add polynomial corrections. These correspond to the first few terms in a multipole-like expansion of the boundary.

3. **This explains why all regularizers agree on gross features.** The bowl shape, the flange extent (~1.5R beyond disk edge), and the depth scale (~0.1–0.2R) are projections onto the well-determined modes. They appear in every solution because the field data forces them. The fine-scale differences between regularizers live entirely in the null space.

4. **Minimum mass is ill-posed for this parameterization.** Unlike the 2D pixel grid (where the density filter provides implicit regularization), the 1D boundary has no natural length scale to prevent the optimizer from exploiting the null space. The "true" minimum-mass boundary with continuous $b(r)$ has zero mass (a measure-zero set of infinitely thin spikes).

5. **The 1% threshold is somewhat arbitrary** but the spectrum drops so steeply (4 orders of magnitude in 6 modes) that any reasonable threshold gives the same count. The gap between mode 6 and mode 7 is the natural break point.

## Files

- `flatearth_3d_svd.py` — SVD analysis and truncated-SVD solution (this analysis)
- `gemini_3d_analytic.py` — spline-parameterized nonlinear solver (predecessor)
- `gemini_2d_topo.py`, `gemini_2d_minmass.py`, `gemini_2d_minperim.py` — pixel-based 2D topology optimization showing regularizer dependence
- `gemini_2d_analytic.py` — 2D Chebyshev boundary solver (requires eps>0)
- `GEMINI.md` — overview of the inverse problem and non-uniqueness theory
