#import "functions.typ": *

#set heading(numbering: "1.")

= Notes

- facebook comments
  - https://www.facebook.com/groups/1295673443855365/posts/8282756768480296/


= Abstract
= Introduction

For nearly 200 years a fierce debate over the true nature of our world has been ongoing between those who believe the Earth is shaped like a sphere/geoid (known as "sphericists"/"globe-heads") and those who think the earth is a flat disk or slab ("flat-earthers"/"flerfs").

One such point of contention between these two groups is the issue of the uniformity of gravity.  While a sphere of constant density naturally has consistent, nadir-pointing gravitational acceleration over its surface, on a disk a person walking from the center towards the edge will experience increasing gravitational resistance as if they are walking uphill, shown in (FIXME: figure).  Extensive gravimetric surveys that show local deviations from standard 1 g acceleration of no more than 0.5% (ε=0.005)(FIXME: cite, FIXME: figure of geoid).

// FIXME - figure gravity anomaly

While not all flat-earthers believe in a gravitational force (e.g. preferring the view that the Earth is accelerating upwards at 1 g), for the sake of argument we assume that the classical model of gravity holds.  We also do not address theories adjacent theories, such as hollow Earth (concave or convex), infinite-plane Earth, cosmic turtles,  or Klein-bottle Earth.

In this manuscript, we show that by relaxing the cylindrical slab assumption slightly, it is possible to achieve a gravity field uniformity consistent with gravimetric measurements over the Earth's surface.
We hope that by constructing these examples of mass distributions,  we can put to rest this specific counter-argument and better understand our (flat) world.

In the following sections, we define the physics that relate mass distribution to gravity vector field at the Earth's surface, propose two extensions to the simple cylindrical slab model of Earth and approaches to numerically evaluate them, and finally show a numerical analysis of these models.

= Problem Formulation

In this section, we

In @physics, we derive the physical model that relates a mass distribution below a disk to the gravitational acceleration on the disk's surface, then simplify this relationship for axially-symmetric mass distributions in @physicsaxial.

// FIXME - cylinder not analytic
In section @cylinder we show an analytic cylindrical solution that meets requirements,

== Physics Forward Model and Problem Statement <physics>

Let $D$ be an origin-centered disk with diameter 1 in the $z=0$ plane and $ρ:ℝ^3→ℝ$ be a non-negative scalar field representing mass density that is 0 for $z>0$.

#figure(
    image("figures/physics1.png", width: 25em),
    caption: [Surface of Earth disk D (in red) in $z=0$ plane with mass distribution $ρ$ below.  $x$ is an observation point on the disk and $x'$ is an arbitrary source point that contributes to the gravity field at $x$]
)

We can write the gravitational field $g : ℝ^3→ℝ^3$ at any point $x$ on the surface $D$ as

// FIXME - assume G=1

$
    g(x) //= integral_(ℝ^3) g(x, x') dif x'
    = integral_(ℝ^3) G ρ(x') / (|x - x'|^2) dot.op (x - x') / (|x - x'|) dif x'
$

// FIXME - units, use -1

Then the problem of finding a mass distribution which produces a uniform downward acceleration $g_0 = (0, 0, -9.81)$ m/s² may be stated

// FIXME - box outline statement?

$
    "Find a density distribution" ρ(x') "which satisfies" g(x) = g_0 "for all" x ∈ D
$

While producing a perfectly uniform downward acceleration would require infinite mass (see @perfect), relaxing the problem to allow deviations in an ε-ball around $g_0$ permits solutions with finite mass.  Indeed, gravimetry measurements show variations in local gravity field on the surface of the Earth up to ~5% (ε=0.005) (FIXME: cite).

// FIXME - epsilon ball

The problem, restated, is

$
    "Find a density distribution" ρ(x') "which satisfies" |g(x) - g_0| ≤ ε|g_0| "for all" x ∈ D
$

// FIXME - cartesian figure here

== Axial Symmetry <physicsaxial>

Assuming that $ρ$ is axially-symmetric around the z-axis, we can write $x$ and $x'$ in cylindrical coordinates without loss of generality as

$
    x = vec(r, 0, 0) "and" x' = vec(r' cos theta', r' sin theta', z') \
$

and reparameterize the density $ρ(x') → ρ(r', z')$

// FIXME - radial figure here

Then the gravity field in cylindrical coordinates is
$
    g_(r)(r) &=
    integral_0^(2 pi) integral_0^infinity integral_(-b(r'))^0
    1 dot.op (r - r' cos theta')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    dif z' r' dif r' dif theta' \

    g_(theta)(r) &= 0 \

    g_(z)(r) &=
    integral_0^(2 pi) integral_0^infinity integral_(-b(r'))^0
    1 dot.op (-z')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    dif z' r' dif r' dif theta' \
$

and the angular component $g_θ$ is 0 due to symmetry.

== Overhang Cylinder Parameterization <cylinder>

== Axially-Symmetric Slab Parameterization <parameterization>

In this section, we define the Axially-Symmetric Slab (ASS) parameterization, which assumes density $ρ$ is connected and unit density between $z=0$ and $z=-b(r')$

#math.equation($
    ρ
$)

= Results <results>


= Conclusion and Future Work

In this work we discuss the issue of gravity uniformity in the flat model of the Earth.  By allowing small deviations in the gravity field from a nadir pointing unit-vector, we show that an axially-symmetric slab of mass can produce a field that is consistent with ground-based gravimetry measurements (FIXME: cite).

// Furthermore, by numerically minimizing slab mass, we demonstrate an inverse relationship between max gravity field deviation and total mass.

// FIXME - drop assumptions
// -

= Acknowledgements

This work was supported in part by the National Sci-Ants Foundation Grant 133769420.

We thank Chester "Chet" Geebeedee for his assistance in validating the derivation of the gravity field equations and Claudia Kody for reducing memory requirements in the PyTorch minimization.


= Appendix

== Symbols

- $x ∈ ℝ^3$ - observation point on surface of disk
- $x' ∈ ℝ^3$ - source point below surface of disk
- $g ∈ ℝ^3$ - gravity vector field on surface of disk

== ASS Gravity Field

First write the gravity field at an observation point $x$ on the disk surface due to a point source of mass $x'$ beneath the disk.

$
    g(x, x') = G rho(x') / (|x - x'|) dot.op (x - x') / (|x - x'|) = rho(x') dot.op (x - x') / (|x - x'|^3)
$

Integrating over all $x' ∈ ℝ$

$
    g(x) = integral_(bb(R)^3) g(x, x') dif x'
    = integral_(bb(R)^3) rho(x') dot.op (x - x') / (|x - x'|^3) dif x'
$

Due to the assumed axial symmetry of $rho$, without loss of generality we can write in cylindrical coordinates:

$
    x = vec(r, 0, 0) "and" x' = vec(r' cos theta', r' sin theta', z') \
    x - x' = vec(r - r' cos theta', -r' sin theta', -z') "and" |x - x'|^2 = r^2 + r'^2 - 2r r' cos theta' + z'^2
$

and reparameterize the density as $ρ(x') → ρ(r', z')$.

Then the gravity field becomes

$
    g(x) = g(r)
    &=
    integral_(theta') integral_(r') integral_(z')
    rho(r', z')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    vec(r - r'cos theta', -r' sin theta', -z')
    dif z' r' dif r' dif theta' \
    // &#gt([substituting flat slab parameterization $b(r')$ of $rho$]) \

    &=
    integral_0^(2 pi) integral_0^infinity integral_(-infinity)^0
    rho(r', z')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    vec(r - r'cos theta', -r' sin theta', -z')
    dif z' r' dif r' dif theta' \
    &#gt([note that the y component is an odd function in $theta'$]) \

    &=
    integral_0^(2 pi) integral_0^infinity integral_(-infinity)^0
    rho(r', z')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    vec(r - r'cos theta', 0, -z')
    dif z' r' dif r' dif theta'
$

Then we are left with only radial and vertical components of the gravity field, as expected from an axially symmetric $rho$

$
    g_(r)(r) &=
    integral_0^(2 pi) integral_0^infinity integral_(-infinity)^0
    rho(r', z') dot.op (r - r' cos theta')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    dif z' r' dif r' dif theta' \

    g_(z)(r) &=
    integral_0^(2 pi) integral_0^infinity integral_(-infinity)^0
    rho(r', z') dot.op (-z')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    dif z' r' dif r' dif theta' \
$

Substituting the ASS parameterization for $ρ$

$
    g_(r)(r) &=
    integral_0^(2 pi) integral_0^infinity integral_(-b(r'))^0
    1 dot.op (r - r' cos theta')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    dif z' r' dif r' dif theta' \

    g_(z)(r) &=
    integral_0^(2 pi) integral_0^infinity integral_(-b(r'))^0
    1 dot.op (-z')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    dif z' r' dif r' dif theta' \
$

As we are doing a numerical optimization of the gravity field, the derivative with respect to $b(r')$ is of interest.  Applying the fundamental theorem of calculus, we get

$
    dif / (dif b(r')) [g_(r)(r)]
    &=
    integral_0^(2 pi) 0 -
    (r - r' cos theta')
    / (r^2 + r'^2 - 2 r r' cos theta' + b(r')^2)^(3/2) (-1)
    r' dif theta' \


    &=
    integral_0^(2 pi)
    (r - r' cos theta')
    / (r^2 + r'^2 - 2 r r' cos theta' + b(r')^2)^(3/2)
    r' dif theta'
$

$
    dif / (dif b(r')) [g_(z)(r)]
    &=
    integral_0^(2 pi) 0 -
    -(-b(r'))
    / (r^2 + r'^2 - 2 r r' cos theta' + b(r')^2)^(3/2) (-1)
    r' dif theta' \


    &=
    integral_0^(2 pi)
    b(r')
    / (r^2 + r'^2 - 2 r r' cos theta' + b(r')^2)^(3/2)
    r' dif theta'
$

== Perfectly Uniform Gravity Field <perfect>

hello

== Rescaling to SI Units
