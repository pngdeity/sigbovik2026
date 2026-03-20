#import "functions.typ": *
#import "template2.typ": *

#show: setup

#title-block(
    title: [Flattening the Discrepancy: \ Gravity-Consistent Model for a Disk-Shaped Earth],
    abstract: none,
    authors: (
        (
            name: "Evan Widloski",
            email: "evan_bovik@widloski.com"
        ),
    ),
    index-terms: ("flat earth", "optimization", "PyTorch", "GPU"),
)

// = Notes

// - facebook comments
//   - https://www.facebook.com/groups/1295673443855365/posts/8282756768480296/

#columns(2, gutter: 12pt)[

= Abstract

The flat Earth model, in which the Earth is a disk or cylindrical slab of uniform density, predicts under classical Newtonian mechanics a gravitational field that varies significantly over the Earth's surface — inconsistent with global gravimetry measurements showing deviations of less than 0.1% from standard 1g acceleration. In this work, we investigate whether allowing variable thickness of the cylindrical slab can reduce the gravity variations. We introduce the Axially-Symmetric Slab (ASS) model, which allows the slab thickness to vary as a function of radius, and pose the problem of finding a density profile that minimizes total mass while keeping field non-uniformity over the disk surface below a chosen threshold. Using a differentiable forward model and numerical optimization, we identify slab profiles that satisfy these constraints. Our results demonstrate that a non-uniform flat Earth mass distribution can produce a gravitational field consistent with deviation observed in gravimetric surveys, offering a quantitative response to one of the primary physical objections to the flat Earth model.

= Introduction

For nearly 200 years a fierce debate over the true nature of our world has been ongoing between those who believe the Earth is shaped like a sphere/geoid (known as "sphericists"/"globe-heads") and those who think the earth is a flat disk or slab ("flat-earthers"/"flerfs").

While early discussions of flat earth were often published as pamphlets steeped in arguments of religious fundamentalism @rowbotham, modern discourse has shifted to venues with significantly more scientific peer review, such as Facebook comments.  Despite the change, significant disagreement occurs between the two communities.

One such point of contention between these two groups is the issue of the uniformity of gravity.  While a sphere of constant density naturally has consistent, nadir-pointing gravitational acceleration over its surface, on a disk a person walking from the center towards the edge will experience increasing gravitational resistance as if they are walking uphill, shown in @intro1.  Extensive gravimetric surveys that show local deviations from standard 1 g acceleration of no more than 0.1% (ε=0.001) @anomaly7, as in @anomalymap.


#figure(
    image("figures/intro1.svg", width: 25em),
    caption: [Gravity field of a cylindrical flat Earth and a spherical Earth]
) <intro1>

#figure(
    image("figures/anomaly_scratch.jpg", width: 25em),
    caption: [Gravity anomaly map from GOCE spacecraft. \ 1000 mGal $approx$ 0.001 g]
) <anomalymap>

While not all flat-earthers believe in a gravitational force (e.g. preferring the view that the Earth is accelerating upwards at 1 g), for the sake of argument we assume that the classical model of gravity holds.  We also do not address theories adjacent to flat Earth, such as hollow Earth (convex or concave), infinite-plane Earth, toroidal Earth, Klein-bottle Earth or cosmic turtles.

In this manuscript, we show that by relaxing the cylindrical slab assumption slightly, it is possible to achieve a gravity field uniformity consistent with gravimetric measurements over the Earth's surface.  We provide some examples found numerically #footnote([Code available at https://github.com/evidlo/sigbovik2026]).
We hope that by constructing these examples of mass distributions,  we can put to rest this specific counter-argument and better understand our (flat) world.

In the following sections, we define the physics that relate mass distribution to gravity vector field at the Earth's surface, propose an extension to the simple cylindrical slab model of Earth and an approach for numerical evaluation, and finally show a solutions that meet gravimetry constraints.

// = Problem Formulation

// In this section, we

// In @physics, we derive the physical model that relates a mass distribution below a disk to the gravitational acceleration on the disk's surface, then simplify this relationship for axially-symmetric mass distributions in @physicsaxial.

// // FIXME - cylinder not analytic
// In section @cylinder we show an analytic cylindrical solution that meets requirements,

= Physics Forward Model and Problem Statement <physics>

In this section we establish the physics of the flat Earth mass distribution problem, then write out a constrained optimization problem that finds a mass distribution meeting the gravity uniformity requirement.

== Arbitrary Mass Distribution

Let $D$ be an origin-centered disk with diameter 1 in the $z=0$ plane and $ρ:ℝ^3→ℝ$ be a non-negative scalar field representing Earth density that is 0 for $z>0$.

#figure(
    image("figures/physics1.svg", width: 25em),
    caption: [Surface of Earth disk D (in red) in $z=0$ plane with mass distribution $ρ$ below.  $x$ is an observation point on the disk and $x'$ is an arbitrary source point that contributes to the gravity field at $x$]
)

We can write the gravitational field $g : ℝ^3→ℝ^3$ at any point $x$ on the surface $D$ as

// FIXME - assume G=1

$
    g(x) //= integral_(ℝ^3) g(x, x') dif x'
    = integral_(ℝ^3) G ρ(x') / (|x - x'|^2) dot.op (x - x') / (|x - x'|) dif x'
$

Then the problem of finding a mass distribution which produces a uniform downward acceleration $g_0 = (0, 0, -1)$ m/s² may be stated

#box(
    inset: 0.5em,
    stroke: 1pt + gray,
    $
        // "Find a density distribution"
        "Find" ρ(x') = arg min_ρ integral_(ℝ^3) ρ(x') dif x' \
        "subject to" |g(x) - g_0| = 0 "for all" x ∈ D
    $
)

We look for a (not necessarily unique) minimum mass solution because this problem is underconstrained (e.g. doubling the distance of a point mass from an observer and quadrupling mass leaves gravitational force unchanged).

By the identity theorem of harmonic functions, the requirement that $|g(x) - g_0| = 0$ on the disk $D$ would require infinite mass in $ρ(x')$ (e.g. an infinite uniform slab). However, relaxing the constraint to allow deviations in an ε-ball around $g_0$ permits solutions with finite mass.  Indeed, gravimetry measurements show variations in local gravity field on the surface of the Earth up to \~0.1% (ε=0.001) @anomaly7.

// FIXME - epsilon ball

The problem, restated, is

#box(
    inset: 0.5em,
    stroke: 1pt + gray,
    $
        // "Find a density distribution"
        "Find" ρ(x') = arg min_ρ integral_(ℝ^3) ρ(x') dif x' \
        "subject to" |g(x) - g_0| ≤ ε|g_0| "for all" x ∈ D
    $
)

// FIXME - cartesian figure here

== Axially-Symmetric Mass Distribution <physicsaxial>

Assuming that $ρ$ is axially-symmetric around the z-axis, we can write $x$ and $x'$ in cylindrical coordinates without loss of generality as

$
    x = vec(r, 0, 0) "and" x' = vec(r' cos theta', r' sin theta', z') \
$

and reparameterize the density $ρ(x') → ρ(r', z')$

// FIXME - radial figure here

Then the gravity field in cylindrical coordinates is
$
    g_(r)(r) &=
    integral_0^(2 pi) integral_0^infinity integral_(-infinity)^0
    (ρ(r', z') dot.op (r - r' cos theta'))
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    dif z' r' dif r' dif theta' \

    g_(theta)(r) &= 0 \

    g_(z)(r) &=
    integral_0^(2 pi) integral_0^infinity integral_(-infinity)^0
    (ρ(r', z') dot.op -z')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    dif z' r' dif r' dif theta' \
$

and the angular component $g_θ$ is 0 due to symmetry (see Appendix @derivation).

// = Mass Distribution Models

// == Overhang Cylinder Parameterization <cylinder>

= Axially-Symmetric Slab Model <parameterization>

In this section we define the Axially-Symmetric Slab (ASS) model for $ρ$, in which a slab of unit density is bounded above by $z'=0$ and below by $z'=-b(r')$.  In this case, the mass distribution is

#math.equation($
    ρ(r', z') := cases(
    1 "if" 0 ≤ z' ≤ -b(r'),
    0 "else",
    )
$)

where $-b(r')$ is a 1D profile that lower-bounds the slab when revolved around the $z$ axis.

// FIXME - illustration of -b(r')

#figure(
    image("figures/ass1.svg"),
    caption: "Example ASS models and associated b(r') profiles"
)


= Results <results>

@minmass_results shows the results from numerically optimizing $b(r)$.

The minimum-mass slab cross-section narrows with increasing field-error tolerance, with the optimal mass ranging from 2.36 (ε = 0.01) to 4.12 (ε = 0.001) in units of the target field strength. The corresponding gravity deviation profiles confirm that the error constraint is satisfied across the disk, with tighter tolerances producing thicker, more concentrated slabs beneath the disk center.

#figure(
    grid(
        columns: (1fr, 1fr, 1fr),
        gutter: 12pt,
        image("figures/figure_0.01.png"),
        image("figures/figure_0.005.png"),
        image("figures/figure_0.001.png"),
    ),
    caption: [],
    placement: bottom,
    scope: "parent",
) <minmass_results>


= Conclusion and Future Work

In this work we discuss the issue of gravity uniformity in the flat model of the Earth.  By allowing small deviations in the gravity field from a nadir pointing unit-vector, we show that an axially-symmetric slab of mass can produce a field that is consistent with global gravimetry surveys.

Potential future directions for research could include

- bounds on total mass required to meet a given ε constraint
- variable density besides $ρ = {0, 1}$
- counteracting inward-pointing gravity field of the cylindrical slab model by spinning the Earth to add centrifugal force

// Furthermore, by numerically minimizing slab mass, we demonstrate an inverse relationship between max gravity field deviation and total mass.

// FIXME - drop assumptions
// -

= Acknowledgements

This work was supported in part by the National Sci-Ants Foundation Grant 133769420.

We thank Chester "Chet" Geebeedee for his assistance in validating the derivation of the gravity field equations and Claudia Kody for reducing memory requirements in the PyTorch minimization.

] // end two-column

= Appendix

== Symbols

- $x ∈ ℝ^3$ - observation point on surface of disk
- $x' ∈ ℝ^3$ - source point below surface of disk
- $g(x) ∈ ℝ^3$ - gravity vector field at a point $x$ on surface of disk

== Axially-Symmetric Gravity Field <derivation>

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

Performing the $theta'$ integral using the half-angle substitution $phi = theta' slash 2$ and writing the denominator as $Delta = M^2(1 - k^2 cos^2 phi)$ where

$
    k = (2 sqrt(r r')) / M, quad
    m^2 = (r - r')^2 + z'^2, quad
    M = sqrt((r + r')^2 + z'^2)
$

yields

$
    integral_0^(2 pi) (r - r' cos theta') / Delta^(3/2) d theta'
    &= 2 / (r M) lr([K(k) - (r'^2 - r^2 + z'^2) / m^2 E(k)]) \

    integral_0^(2 pi) 1 / Delta^(3/2) dif theta'
    &= (4 E(k)) / (m^2 M)
$

where $K(k)$ and $E(k)$ are the complete elliptic integrals of the first and second kind. Substituting:

$
    g_(r)(r) &=
    integral_0^infinity integral_(-infinity)^0
    rho(r', z') 2 / (r M)
    lr([K(k) - (r'^2 - r^2 + z'^2) / m^2 E(k)])
    dif z' r' dif r' \

    g_(z)(r) &=
    integral_0^infinity integral_(-infinity)^0
    rho(r', z') (-z' 4 E(k)) / (m^2 M)
    dif z' r' dif r'
$

== ASS Gravity Field

Substituting the ASS model for $ρ$ into $g_r$ and $g_z$ from the previous section

$
    g_(r)(r) &=
    integral_0^(2 pi) integral_0^infinity integral_(-b(r'))^0
    1 dot.op (r - r' cos theta')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    dif z' r' dif r' dif theta' \

    &=
    integral_0^infinity integral_(-b(r'))^0
    1 dot.op 2 / (r M)
    lr([K(k) - (r'^2 - r^2 + z'^2) / m^2 E(k)])
    dif z' r' dif r' \

    g_(z)(r) &=
    integral_0^(2 pi) integral_0^infinity integral_(-b(r'))^0
    1 dot.op (-z')
    / (r^2 + r'^2 - 2 r r' cos theta' + z'^2)^(3/2)
    dif z' r' dif r' dif theta' \

    &=
    integral_0^infinity integral_(-b(r'))^0
    1 dot.op (-z' 4 E(k)) / (m^2 M)
    dif z' r' dif r'
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
    r' dif theta' \

    &=
    r' 2 / (r M) lr([K(k) - (r'^2 - r^2 + b(r')^2) / m^2 E(k)])
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
    r' dif theta' \

    &=
    (4 r' b(r') E(k)) / (m^2 M)
$

where $k$, $m$, $M$ are as defined above, evaluated at $z' = b(r')$.

This eliminates the $r'$ and $z'$ integrals and greatly accelerates optimization as compared to autograd through the 2 numerical integrations involved in computing $g_r$ and $g_z$.

// == Rescaling to SI Units
// FIXME

#bibliography("refs.bib")