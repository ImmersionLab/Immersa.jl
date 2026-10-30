#=
Builders: resolved `CaseConfig` → Immersa objects (grid, body, problem, solver).
=#

"""
    BodyLayout

Where each case-file body lives in the merged marker arrays.
`names[k]`'s markers are `ranges[k]`; `x_ref`/`ds` are the reference markers.
"""
struct BodyLayout{N}
    names::Vector{String}
    ranges::Vector{UnitRange{Int}}
    x_ref::Vector{SVector{N,Float64}}
    ds::Vector{Float64}
end

"""
    CaseSetup

Everything [`build_case`](@ref) produces: the config, the problem, the body layout,
the time-step count, and the solver constructor arguments.
"""
struct CaseSetup{C<:CaseConfig,P<:IBProblem,L<:BodyLayout}
    cfg::C
    prob::P
    layout::L
    backend::Any
    n_steps::Int
end

function make_backend(name)
    name == "CPU" && return CPU()
    pkg, ctor = if name == "CUDA"
        Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"), :CUDABackend
    else
        Base.PkgId(Base.UUID("21141c5a-9bdb-4563-92ae-f87d6854732e"), "AMDGPU"), :ROCBackend
    end
    mod = try
        Base.require(pkg)
    catch
        config_error("[case] backend = \"$name\" needs the $(pkg.name) package installed in the active environment")
    end
    Base.invokelatest(getfield(mod, ctor))
end

function build_grid(g, dim)
    if g.type == "uniform"
        lo, hi = g.limits[:, 1], g.limits[:, 2]
        n = round.(Int, (hi .- lo) ./ g.h)
        Grid(; h=g.h, n=SVector{dim}(n), x0=SVector{dim}(lo), levels=g.levels)
    else
        StretchedGrid(;
            dx_min=g.h,
            core=SMatrix{dim,2}(g.core),
            growth=g.growth,
            extent=SMatrix{dim,2}(g.extent),
            max_dx=g.max_dx,
        )
    end
end

function build_bodies(cfg::CaseConfig, h)
    N = cfg.dim
    vN = Val(N)
    names, ranges, parts = String[], UnitRange{Int}[], AbstractPrescribedBody[]
    x_all, ds_all = SVector{N,Float64}[], Float64[]
    for b in cfg.bodies
        x, ds = body_geometry(b, h, N)
        r = (length(x_all) + 1):(length(x_all) + length(x))
        push!(names, b.name)
        push!(ranges, r)
        updater = make_updater(b.motion, vN, x, "[body \"$(b.name)\"]")
        push!(parts, isnothing(updater) ? StaticBody(x, ds) : MovingBody(x, ds, BodyMotion(updater, x)))
        append!(x_all, x)
        append!(ds_all, ds)
    end
    layout = BodyLayout{N}(names, ranges, x_all, ds_all)

    b1 = first(cfg.bodies)
    body = if b1.model == "deformable"
        st, n = b1.structure, length(x_all)
        bcs = st.clamped ? [StructureBC{Float64}(i, (_, _, _) -> 0.0) for i in 1:3] :
              StructureBC{Float64}[]
        GeometricNonlinearBody(;
            xref=x_all,
            ds0=ds_all,
            m=fill(st.mass, n),
            kb=fill(st.bending_stiffness, n),
            ke=fill(st.extensional_stiffness, n),
            bcs,
        )
    elseif length(parts) == 1
        only(parts)
    else
        # Markers are stored back to back in [[body]] order, matching `ranges`.
        GroupedPrescribedBody(Tuple(parts))
    end
    body, layout
end

"""
    build_case(cfg::CaseConfig) -> CaseSetup
    build_case(path) -> CaseSetup

Build the grid, bodies and `IBProblem` described by a case (no solver yet; see
[`build_solver`](@ref)).
"""
build_case(path::AbstractString) = build_case(load_case(path))

function build_case(cfg::CaseConfig)
    backend = make_backend(cfg.case.backend)
    grid = build_grid(cfg.grid, cfg.dim)
    body, layout = build_bodies(cfg, gridstep(grid))
    u0 = UniformFlow(make_freestream(cfg.flow, Val(cfg.dim)))
    formulation = cfg.solver.formulation == "IBPM" ? IBPM() : FastIBPM()
    prob = IBProblem(grid, body, cfg.flow.Re, u0, formulation)
    n_steps = 1 + round(Int, cfg.solver.t_final / cfg.solver.dt)
    CaseSetup(cfg, prob, layout, backend, n_steps)
end

"""
    build_solver(setup::CaseSetup) -> CNAB

Construct the CNAB time integrator and apply the initial condition (and restart
file, if any).
"""
function build_solver(setup::CaseSetup)
    (; cfg, prob, backend) = setup
    s = cfg.solver
    delta = s.delta == "Yang3S" ? DeltaYang3S() :
            s.delta == "Roma"   ? DeltaRoma()   : DeltaYang3S2()
    deformable = prob.body isa GeometricNonlinearBody
    # Same BiCGStab settings as the original flag.jl example.
    coupler_args = deformable ? (; bicgstabl_args=(;)) : (;)
    sol = CNAB(
        prob;
        dt=s.dt,
        delta,
        backend,
        n_step=s.ab_order,
        coupler_args,
        precond=Symbol(s.preconditioner),
        coupling_tol=s.coupling_tol,
    )

    if deformable && cfg.bodies[1].structure.initial_angle != 0
        θ = cfg.bodies[1].structure.initial_angle
        g = cfg.bodies[1].geom
        χ = sol.coupler.state.χ
        x = range(0, g.length, length(setup.layout.x_ref))
        @. χ[2:3:end] = sin(θ) * x
        @. χ[3:3:end] = θ
    end

    apply_initial_condition!(sol, cfg.initial_condition)

    if !isnothing(s.restart_from)
        open(io -> load!(io, sol), s.restart_from)
        @info "Restarted from $(s.restart_from) at step $(sol.i), t = $(sol.t)"
    end
    sol
end

function apply_initial_condition!(sol, ic)
    ic.type == "none" && return sol
    if ic.type == "vortex_blob"
        grid = sol.prob.grid
        c = SVector{2}(ic.center)
        (; radius, amplitude) = ic
        map!(sol.state.ω[1][3], CartesianIndices(sol.state.ω[1][3])) do I
            x = coord(grid, Loc_ω(3), I)
            amplitude * (1 - clamp(norm(x - c) / radius, 0, 1))
        end
        apply_vorticity!(sol)
    end
    sol
end
