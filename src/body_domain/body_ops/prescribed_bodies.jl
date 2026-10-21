"""
This file defines types and methods for handling bodies whose motion is defined ahead of time.

A prescribed body can be static or moving, but cannot be fully coupled or deformable.

- `AbstractPrescribedBody` (a subtype of `AbstractBody`) is the superclass for all predefined-motion bodies.
- `AbstractStaticBody` is a special case where the motion is zero everywhere (i.e., static).

Hierarchy sketch:

  AbstractBody
    +- AbstractPrescribedBody
      +- AbstractStaticBody
        +- NothingBody         (dummy)
        +- StaticBody          (real, immobile body)

"""

"""
    AbstractPrescribedBody <: AbstractBody

Abstract supertype for all bodies whose motion is fully prescribed (not coupled).

Subtypes include `AbstractStaticBody` (zero motion) and `AbstractMovingBody`
(time-dependent prescribed motion).
"""
abstract type AbstractPrescribedBody <: AbstractBody end

"""
    AbstractStaticBody <: AbstractPrescribedBody

Abstract supertype for bodies with zero motion. Concrete subtypes: `NothingBody`,
`StaticBody`.
"""
abstract type AbstractStaticBody <: AbstractPrescribedBody end

"""
    AbstractMovingBody <: AbstractPrescribedBody

Abstract supertype for bodies with prescribed time-dependent motion. Concrete
subtype: `MovingBody`.
"""
abstract type AbstractMovingBody <: AbstractPrescribedBody end

"""
    NothingBody{}

The NothingBody struct defines a trivial, empty static body used as a placeholder when no physical body is present.

# Related functions
Functions associated with NothingBody perform no operations:
- `point_count(::NothingBody)`: returns 0 (no points).
- `init_body_points!(::BodyPoints, ::NothingBody)`: does nothing.
- `update_body_points!(::BodyPoints, ::NothingBody, i, t)`: does nothing.
"""
struct NothingBody <: AbstractStaticBody end

point_count(::NothingBody) = 0
init_body_points!(::BodyPoints, ::NothingBody) = nothing
update_body_points!(::BodyPoints, ::NothingBody, i, t) = nothing

"""
    StaticBody{N,T,S<:AbstractVector{T},A<:AbstractVector{SVector{N,T}},U}

A real, immobile body defined by fixed positions and spacings, with an optional
velocity callback for actuated surfaces.

# Fields
- `x::A` : Vector of positions (`SVector{N,T}`) in N-dimensional space.
- `ds::S` : Vector of weights or spacings associated with points.
- `u::U`  : Velocity callback `(u, i, t) -> nothing`, called at each step. Defaults to a no-op.

# Type parameters
- `N` : Spatial dimension.
- `T` : Scalar type (e.g., `Float64`).
- `S` : Type of vector for `ds`.
- `A` : Type of vector for `x`.
- `U` : Type of the velocity callback.

# Constructor
    StaticBody(x, ds)
    StaticBody(x, ds, u)

When `u` is omitted, a no-op callback is used.

# Related functions
- `point_count(body)` : Returns the number of points on the body.
- `init_body_points!(points, body)` : Initialises `BodyPoints` with the static body's positions, zero velocities, and spacings.
- `update_body_points!(points, body, i, t)` : No-op; static bodies do not move.
"""
struct StaticBody{N,T,S<:AbstractVector{T},A<:AbstractVector{SVector{N,T}},U} <:
       AbstractStaticBody
    x::A
    ds::S
    u::U
end

StaticBody(x, ds) = StaticBody(x, ds, (u, i, t) -> nothing)

# TODO(known issue): the velocity callback `u` is stored but never called —
# `update_body_points!(::BodyPoints, ::StaticBody, i, t)` is a no-op, so marker
# velocities stay zero and surface actuation through `StaticBody` has no effect
# (examples/plate_actuation.jl has been running an un-actuated plate). Case files
# avoid this by driving actuation through a `MovingBody` with fixed positions.
# Fix: call `body.u(points.u, i, t)` in that method — note that `B` would then stay
# constant (positions don't move), so the precomputed static `B⁻¹` remains valid.

"""
    point_count(body::StaticBody) -> Int

Return the number of immersed-boundary points on a `StaticBody`.
"""
point_count(body::StaticBody) = length(body.x)

"""
    init_body_points!(points::BodyPoints, body::StaticBody{N,T})

Copy the static body's positions and spacings into `points` and set all velocities
to zero.
"""
function init_body_points!(points::BodyPoints, body::StaticBody{N,T}) where {N,T}
    points.x .= body.x
    points.u .= (zero(SVector{N,T}),)
    points.ds[:] = body.ds
end

"""
    PoseTwist2D{T}

Describes a 2-D rigid-body pose (position + orientation) and its time derivative
(translational and angular velocity).

# Fields
- `c::SVector{2,T}` : Centre-of-mass translation.
- `θ::T`            : Rotation angle.
- `ċ::SVector{2,T}` : Translational velocity.
- `θ̇::T`           : Angular velocity.
"""
@kwdef struct PoseTwist2D{T}
    c::SVector{2,T} # translation
    θ::T            # rotation angle
    ċ::SVector{2,T} # translation velocity
    θ̇::T           # angular velocity
end

"""
    MovingBody{N,T,S,A,F} <: AbstractMovingBody

A body with prescribed time-dependent motion defined by a user-supplied callback.

# Fields
- `x_ref::A`  : Reference (initial) positions (`Vector{SVector{N,T}}`).
- `ds::S`     : Weights or spacings associated with points.
- `motion!::F`: Callback `(x, u, i, t) -> nothing` that overwrites positions `x` and
  velocities `u` at step `i` and time `t`.

# Constructor
    MovingBody(x_ref, ds, motion!)

# Related functions
- `point_count(body)` : Returns the number of points.
- `init_body_points!(points, body)` : Initialises `BodyPoints` from the reference shape with zero velocity.
- `update_body_points!(points, body, i, t)` : Calls `motion!` to update positions and velocities.
"""
struct MovingBody{N,T,S<:AbstractVector{T},A<:AbstractVector{SVector{N,T}},F} <:
       AbstractMovingBody
    x_ref::A          # reference positions
    ds::S             # weights or spacings
    motion!::F        # function defining
    function MovingBody(
        x_ref::AbstractVector{SVector{N,T}}, ds::AbstractVector{T}, motion!::F
    ) where {N,T,F}
        new{N,T,typeof(ds),typeof(x_ref),F}(x_ref, ds, motion!)
    end
end

"""
    point_count(body::MovingBody) -> Int

Return the number of immersed-boundary points on a `MovingBody`.
"""
point_count(b::MovingBody) = length(b.x_ref)

"""
    init_body_points!(points::BodyPoints, body::MovingBody{N,T})

Copy the reference positions and spacings into `points` and set all velocities
to zero.
"""
function init_body_points!(points::BodyPoints, body::MovingBody{N,T}) where {N,T}
    points.x .= body.x_ref
    points.u .= (zero(SVector{N,T}),)
    points.ds[:] = body.ds
    return nothing
end

"""
    update_body_points!(points::BodyPoints, body::MovingBody, i, t)

Invoke the body's `motion!` callback to overwrite positions and velocities at
step `i` and time `t`.
"""
function update_body_points!(pts::BodyPoints{N,T}, b::MovingBody{N,T}, i, t) where {N,T}
    b.motion!(pts.x, pts.u, i, t)
    nothing
end

"""
    update_body_points!(::BodyPoints, ::StaticBody, i, t)

No-op for static bodies — positions and velocities are unchanged.
"""
update_body_points!(::BodyPoints, ::StaticBody, i, t) = nothing

"""
    GroupedPrescribedBody{T<:Tuple{Vararg{AbstractPrescribedBody}}}

A composite body that groups multiple `AbstractPrescribedBody` instances into a
single body interface. The markers of the sub-bodies are stored back to back, in
tuple order: sub-body `k` owns the slice `body_ranges(body)[k]` of the marker arrays,
and every operation on it acts on that slice only.

A group in which **every** sub-body is static (`AbstractStaticBody`) is treated as
static by the solver (precomputed `B⁻¹`, no per-step regularization update); a group
containing at least one moving body uses the iterative, per-step path.

# Fields
- `bodies::T` : Tuple of prescribed bodies.

# Related functions
- `point_count(body)` : Returns the sum of point counts across all sub-bodies.
- `body_ranges(body)` : Marker index range of each sub-body.
- `init_body_points!(points, body)` : Initialises each sub-body's slice of `points`.
- `update_body_points!(points, body, i, t)` : Updates each sub-body's slice of `points`.
"""
struct GroupedPrescribedBody{T<:Tuple{Vararg{AbstractPrescribedBody}}} <:
       AbstractPrescribedBody
    bodies::T
end

"""
    StaticGroupedBody

A [`GroupedPrescribedBody`](@ref) whose sub-bodies are all static.
"""
const StaticGroupedBody = GroupedPrescribedBody{<:Tuple{Vararg{AbstractStaticBody}}}

"""
    point_count(body::GroupedPrescribedBody) -> Int

Return the total number of immersed-boundary points across all sub-bodies.
"""
point_count(body::GroupedPrescribedBody) = sum(point_count, body.bodies)

"""
    body_ranges(body::GroupedPrescribedBody) -> Tuple of UnitRange

Marker index range of each sub-body in the merged marker arrays.
"""
function body_ranges(body::GroupedPrescribedBody)
    stops = accumulate(+, map(point_count, body.bodies))
    map((b, s) -> (s - point_count(b) + 1):s, body.bodies, stops)
end

"""
    init_body_points!(points::BodyPoints, body::GroupedPrescribedBody)

Initialise each sub-body on its own slice of `points`.
"""
function init_body_points!(points::BodyPoints, body::GroupedPrescribedBody)
    foreach((b, r) -> init_body_points!(view(points, r), b), body.bodies, body_ranges(body))
end

"""
    update_body_points!(points::BodyPoints, body::GroupedPrescribedBody, i, t)

Update each sub-body on its own slice of `points` at step `i` and time `t`.
"""
function update_body_points!(points::BodyPoints, body::GroupedPrescribedBody, i, t)
    foreach(
        (b, r) -> update_body_points!(view(points, r), b, i, t), body.bodies, body_ranges(body)
    )
end

