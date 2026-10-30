#=
Prescribed motion for case files.

Every moving body gets an *updater* `f(x, u, x_ref, t)` that overwrites the marker
positions `x` and velocities `u` of that body (views into host arrays) from its
reference positions `x_ref`. Updaters come from:

  - a built-in rigid motion (`translate`, `oscillate`, `pitch_heave`), or
  - a user file (`type = "user"`) defining ONE of
      rigid_motion(t[, params])            -> (; center, angle, velocity, angular_velocity)
      motion!(x, u, x_ref, t[, params])    -> general prescribed motion / deformation
      surface_velocity!(u, x_ref, t[, params]) -> fixed body, actuated surface velocity
    where `params` is the [body.motion.parameters] table as a NamedTuple.

Each moving body becomes a `MovingBody` whose callback (`BodyMotion`) runs the
updater on host arrays and copies the result to the solver's marker arrays (which
may live on a GPU). Several bodies are combined with `GroupedPrescribedBody`.
=#

# ---------------------------------------------------------------------------
# Rigid motion: displacement `c` and rotation `θ` about `pivot`, with rates.
#   x = pivot + c + R(θ) (x_ref - pivot),     u = ċ + θ̇ ẑ × R(θ) (x_ref - pivot)
# ---------------------------------------------------------------------------

struct RigidMotion{F,N}
    pose::F                 # t -> (c, θ, ċ, θ̇)
    pivot::SVector{N,Float64}
end

function (m::RigidMotion{F,2})(x, u, x_ref, t) where {F}
    c, θ, ċ, θ̇ = m.pose(t)
    if iszero(θ) && iszero(θ̇)  # pure translation: exact x_ref + c, no pivot round-off
        @inbounds for j in eachindex(x)
            x[j] = x_ref[j] + c
            u[j] = ċ
        end
        return nothing
    end
    R = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]
    @inbounds for j in eachindex(x)
        r = R * (x_ref[j] - m.pivot)
        x[j] = m.pivot + c + r
        u[j] = ċ + θ̇ * SA[-r[2], r[1]]
    end
    nothing
end

# 3D: translation only for now (rotation comes with the 3D work).
function (m::RigidMotion{F,3})(x, u, x_ref, t) where {F}
    c, θ, ċ, θ̇ = m.pose(t)
    (iszero(θ) && iszero(θ̇)) || error("rigid rotation is not yet supported in 3D")
    @inbounds for j in eachindex(x)
        x[j] = x_ref[j] + c
        u[j] = ċ
    end
    nothing
end

centroid(x) = sum(x) / length(x)

function builtin_pose(m, ::Val{N}) where {N}
    if m.type == "translate"
        v = SVector{N}(m.velocity)
        t -> (v * t, 0.0, v, 0.0)
    elseif m.type == "oscillate"
        A, ω, φ = SVector{N}(m.amplitude), 2π * m.frequency, m.phase
        t -> (A * sin(ω * t + φ), 0.0, A * ω * cos(ω * t + φ), 0.0)
    elseif m.type == "pitch_heave"
        H, Θ, ω, φ = m.heave_amplitude, m.pitch_amplitude, 2π * m.frequency, m.phase
        t -> (SA[0.0, H * sin(ω * t)], Θ * sin(ω * t + φ),
              SA[0.0, H * ω * cos(ω * t)], Θ * ω * cos(ω * t + φ))
    end
end

# ---------------------------------------------------------------------------
# User files
# ---------------------------------------------------------------------------

"""
    load_user_file(path) -> Module

Evaluate a user `.jl` file in a fresh module (with `StaticArrays` and
`LinearAlgebra` available) and return that module.
"""
function load_user_file(path)
    m = Module(Symbol("UserFile_", splitext(basename(path))[1]))
    Core.eval(m, :(using StaticArrays, LinearAlgebra))
    try
        Base.include(m, path)
    catch e
        config_error("error while loading user file $path:\n" * sprint(showerror, e))
    end
    m
end

# The file's bindings are newer than this code's world: read them via invokelatest.
user_function(m::Module, name) =
    Base.invokelatest(isdefined, m, name) ? Base.invokelatest(getglobal, m, name) : nothing

# Call a user function defined after this code was compiled (needs the latest
# world). If it accepts a trailing `params` argument, pass it.
function user_caller(f, params, args...)
    takes_params = Base.invokelatest(applicable, f, args..., params)
    takes_params ? (a...) -> Base.invokelatest(f, a..., params) :
                   (a...) -> Base.invokelatest(f, a...)
end

function user_updater(m, ::Val{N}, x_ref, where_) where {N}
    mod = load_user_file(m.file)
    fns = [(n, user_function(mod, n)) for n in (:rigid_motion, :motion!, :surface_velocity!)]
    defined = [(n, f) for (n, f) in fns if !isnothing(f)]
    length(defined) == 1 || config_error(
        "$where_: $(m.file) must define exactly one of `rigid_motion`, `motion!`, " *
        "`surface_velocity!` (found: $(isempty(defined) ? "none" : join(first.(defined), ", ")))",
    )
    name, f = only(defined)
    zN = zero(SVector{N,Float64})
    if name === :rigid_motion
        call = user_caller(f, m.params, 0.0)
        pose = function (t)
            nt = call(t)
            nt isa NamedTuple || error("rigid_motion must return a NamedTuple, got $(typeof(nt))")
            (SVector{N,Float64}(get(nt, :center, zN)), Float64(get(nt, :angle, 0.0)),
             SVector{N,Float64}(get(nt, :velocity, zN)), Float64(get(nt, :angular_velocity, 0.0)))
        end
        pose(0.0)  # fail early on a malformed return value
        RigidMotion(pose, SVector{N,Float64}(something(m.pivot, centroid(x_ref))))
    elseif name === :motion!
        call = user_caller(f, m.params, similar(x_ref), similar(x_ref), x_ref, 0.0)
        (x, u, xr, t) -> (call(x, u, xr, t); nothing)
    else # surface_velocity!: positions stay at x_ref
        call = user_caller(f, m.params, similar(x_ref), x_ref, 0.0)
        function (x, u, xr, t)
            copyto!(x, xr)
            call(u, xr, t)
            nothing
        end
    end
end

"""
    make_updater(motion, Val(N), x_ref, where_)

Return the updater `f(x, u, x_ref, t)` for one body, or `nothing` for a static body.
"""
function make_updater(m, vN::Val{N}, x_ref, where_) where {N}
    isnothing(m) && return nothing
    if m.type == "user"
        user_updater(m, vN, x_ref, where_)
    else
        pivot = SVector{N,Float64}(something(get(m, :pivot, nothing), centroid(x_ref)))
        RigidMotion(builtin_pose(m, vN), pivot)
    end
end

# ---------------------------------------------------------------------------
# Updater → MovingBody callback
# ---------------------------------------------------------------------------

struct BodyMotion{F,N,T}
    updater::F
    x_ref::Vector{SVector{N,T}}
    xh::Vector{SVector{N,T}}    # host work arrays (marker arrays may live on a GPU)
    uh::Vector{SVector{N,T}}
end

BodyMotion(updater, x_ref) = BodyMotion(updater, x_ref, copy(x_ref), zero(x_ref))

# Signature required by `MovingBody`: (x, u, i, t) -> nothing.
function (m::BodyMotion)(x, u, _, t)
    m.updater(m.xh, m.uh, m.x_ref, t)
    copyto!(x, m.xh)
    copyto!(u, m.uh)
    nothing
end

# ---------------------------------------------------------------------------
# Free stream
# ---------------------------------------------------------------------------

function make_freestream(flow, ::Val{N}) where {N}
    if !isnothing(flow.file)
        mod = load_user_file(flow.file)
        f = user_function(mod, :freestream)
        isnothing(f) && config_error("[flow] $(flow.file) must define `freestream(t)`")
        return t -> SVector{N,Float64}(Base.invokelatest(f, t))
    end
    U = SVector{N}(flow.freestream)
    τ = flow.ramp_time
    τ > 0 || return t -> U
    t -> (s = clamp(t / τ, 0, 1); s * (2 - s) * U)  # smooth start, as in flag.jl
end
