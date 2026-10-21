#=
Driver: run a case file end to end.

    julia> using Immersa
    julia> run_case("examples/cylinder/case.toml")

or from the shell (see bin/immersa):

    $ immersa examples/cylinder/case.toml
    $ immersa --check examples/cylinder/case.toml   # validate + summary, no run
=#

"""
    run_case(path; check=false) -> CNAB

Load, validate, build and run the case at `path` (a `case.toml` or a folder
containing one). With `check=true`, stop after printing the setup summary.
Returns the final solver state (or `nothing` when `check=true`).
"""
function run_case(path::AbstractString; check::Bool=false)
    cfg = load_case(path)
    setup = build_case(cfg)
    print_summary(stdout, setup)
    check && return nothing

    t_build = time()
    sol = build_solver(setup)
    @printf("Solver ready in %.1f s\n\n", time() - t_build)

    out = open_output(setup, sol)
    o = cfg.output
    t0 = time()
    try
        while sol.i < setup.n_steps
            step!(sol)
            write_step!(out, sol)
            if o.log_every > 0 && (sol.i % o.log_every == 0 || sol.i == setup.n_steps)
                log_step(sol, setup, time() - t0)
            end
            if o.restart_every > 0 && sol.i % o.restart_every == 0
                write_restart(o.directory, sol)
            end
        end
        o.restart_every > 0 && write_restart(o.directory, sol)
    finally
        close_output(out)
    end
    @printf("\nDone: %d steps in %.1f s. Output in %s\n", setup.n_steps, time() - t0, o.directory)
    sol
end

function log_step(sol, setup, wall)
    F = surface_force_sum(sol)
    ref = setup.cfg.reference
    C = F / (0.5 * ref.velocity^2 * ref.length^(length(F) - 1))
    @printf("step %8d/%d | t = %9.4f | wall %8.1f s | C = %s\n",
        sol.i, setup.n_steps, sol.t, wall, join((@sprintf("%+.4f", c) for c in C), " "))
end

function print_summary(io::IO, setup::CaseSetup)
    (; cfg, prob, layout, n_steps) = setup
    grid = prob.grid
    h = gridstep(grid)
    s = cfg.solver
    line = "-"^64
    println(io, line)
    println(io, " Immersa case: ", cfg.case.name)
    println(io, " file:         ", cfg.file)
    println(io, line)
    @printf(io, " %-16s %dD, Re = %g, freestream = %s%s\n", "Flow", cfg.dim, cfg.flow.Re,
        cfg.flow.freestream, isnothing(cfg.flow.file) ? "" : " (from $(basename(cfg.flow.file)))")
    if grid isa Grid
        @printf(io, " %-16s uniform, h = %g, cells = %s, levels = %d\n", "Grid", h,
            join(grid.n, "×"), grid.levels)
    else
        @printf(io, " %-16s stretched, h_min = %g, cells = %s\n", "Grid", h, join(grid.n, "×"))
    end
    for (b, name, r) in zip(cfg.bodies, layout.names, layout.ranges)
        ds = layout.ds[r]
        motion = isnothing(b.motion) ? (b.model == "deformable" ? "deformable" : "static") :
                 b.motion.type == "user" ? "user ($(basename(b.motion.file)))" : b.motion.type
        @printf(io, " %-16s %s, %d markers, ds/h = %.2f, %s\n", "Body \"$name\"",
            b.shape, length(r), sum(ds) / length(ds) / h, motion)
    end
    @printf(io, " %-16s %s, dt = %g, t_final = %g (%d steps), delta = %s\n", "Solver",
        s.formulation, s.dt, s.t_final, n_steps, s.delta)
    U = maximum(norm, (cfg.flow.freestream, [cfg.reference.velocity]))
    @printf(io, " %-16s U·dt/h = %.3f\n", "CFL (approx)", U * s.dt / h)
    cfg.initial_condition.type == "none" ||
        @printf(io, " %-16s %s\n", "Initial cond.", cfg.initial_condition.type)
    o = cfg.output
    @printf(io, " %-16s %s (history every %d, snapshots every %s)\n", "Output", o.directory,
        o.history_every, o.snapshot_every == 0 ? "never" : string(o.snapshot_every))
    println(io, line)
end

"""
    main(args) -> Int

Command-line entry point used by `bin/immersa`. Returns a process exit code.
"""
function main(args::Vector{String}=ARGS)
    usage = """
    usage: immersa [--check] <case.toml | case_folder>

      --check   validate the case file and print the setup summary, without running
    """
    check = "--check" in args
    rest = filter(a -> a != "--check", args)
    if any(a -> a in ("-h", "--help"), rest) || length(rest) != 1
        print(usage)
        return length(rest) == 1 ? 0 : 1
    end
    try
        run_case(only(rest); check)
        return 0
    catch e
        e isa CaseConfigError || rethrow()
        println(stderr, sprint(showerror, e))
        return 2
    end
end
