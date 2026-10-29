#=
Case output.

  <output>/history.csv   forces and force coefficients per body (+ total), every
                         `history_every` steps — readable by any tool. Plates with
                         `track_tip = true` also get their free-end position,
                         `<name>_tip_x`, `<name>_tip_y`.
  <output>/solution.h5   same layout as the original example scripts, so the
                         scripts in `examples/post_proc` work unchanged:
                           body/points, body/lengths            reference markers
                           all/t, all/Cd, all/Cl, all/force     total-force history
                           all/x_tip[_<name>]                   tracked plate tips
                           snapshots/t, omega, ux, uy, [p],     field snapshots (2D)
                           x_coords, y_coords, body_points
                         In 3D the snapshots hold every component on every level,
                         each on its own staggered points:
                           snapshots/omega_x, omega_y, omega_z, ux, uy, uz
                             (n1, n2, n3, levels, n_snapshot)
                           snapshots/coords/<field>/x, y, z     (n_d, levels)
  <output>/restart_<step>.bin  checkpoints (`restart_every`), for [solver] restart_from
  <output>/case.toml     copy of the case file that produced the run
=#

mutable struct CaseOutput{N}
    h5::HDF5.File
    history::IO
    layout::BodyLayout{N}
    i_first::Int            # first step written by this run (> 1 after a restart)
    n_steps::Int
    history_every::Int
    snapshot_every::Int
    fields::Vector{String}
    coef_scale::Float64     # 1 / (½ U² A)
    f_host::Vector{SVector{N,Float64}}
    ω_buf::Any              # IBPM vorticity work array
    tips::Vector{Pair{String,Int}}  # body name => tip marker (bodies with track_tip = true)
end

# The tip of a plate is its free end: the last marker (a flag is clamped at its first).
tracked_tips(cfg, layout) =
    [name => last(r) for (b, name, r) in zip(cfg.bodies, layout.names, layout.ranges)
     if b.track_tip]

# One tracked body → `all/x_tip` (the name the original flag.jl used); several →
# `all/x_tip_<name>`.
tip_dataset_names(tips) =
    length(tips) == 1 ? ["x_tip"] : ["x_tip_$name" for (name, _) in tips]

# Steps i ∈ i_first:n_steps written every `every` steps, counted from step 1 like
# the original examples (i_all[1:snapshot_freq:end]).
_selected(i, every) = every > 0 && (i - 1) % every == 0
_count_selected(i_first, n, every) = every > 0 ? count(i -> _selected(i, every), i_first:n) : 0
_slot(i, i_first, every) = _count_selected(i_first, i, every)

function open_output(setup::CaseSetup, sol::CNAB{N,T}) where {N,T}
    (; cfg, layout, n_steps) = setup
    o = cfg.output
    mkpath(o.directory)
    h5path = joinpath(o.directory, "solution.h5")
    restarting = !isnothing(cfg.solver.restart_from)
    if isfile(h5path) && !o.overwrite
        config_error("[output] $h5path exists and overwrite = false")
    end
    cp(cfg.file, joinpath(o.directory, "case.toml"); force=true)

    i_first = sol.i + 1
    n_hist = _count_selected(i_first, n_steps, o.history_every)
    n_snap = _count_selected(i_first, n_steps, o.snapshot_every)

    h5 = h5open(h5path, "w")
    attributes(h5)["case_file"] = cfg.file
    attributes(h5)["first_step"] = i_first

    bg = create_group(h5, "body")
    bg["points"] = reduce(hcat, layout.x_ref)
    bg["lengths"] = layout.ds
    bg["names"] = layout.names
    bg["ranges"] = reduce(hcat, [[first(r), last(r)] for r in layout.ranges])

    ag = create_group(h5, "all")
    create_dataset(ag, "t", T, (n_hist,))
    create_dataset(ag, "Cd", T, (n_hist,))
    create_dataset(ag, "Cl", T, (n_hist,))
    create_dataset(ag, "force", T, (N, n_hist))
    tips = tracked_tips(cfg, layout)
    foreach(d -> create_dataset(ag, d, T, (N, n_hist)), tip_dataset_names(tips))

    ω_buf = nothing
    if n_snap > 0
        sg = create_group(h5, "snapshots")
        create_dataset(sg, "t", T, (n_snap,))
        create_dataset(sg, "body_points", T, (N, length(layout.x_ref), n_snap))
        ω_buf = create_field_datasets!(sg, sol, o.fields, n_snap)
    end

    hist_path = joinpath(o.directory, "history.csv")
    history = if restarting && isfile(hist_path)
        open(hist_path, "a")
    else
        io = open(hist_path, "w")
        println(io, history_header(layout, N, tips))
        io
    end

    ref = cfg.reference
    coef_scale = 1 / (0.5 * ref.velocity^2 * ref.area)
    f_host = zeros(SVector{N,Float64}, length(layout.x_ref))
    CaseOutput{N}(
        h5, history, layout, i_first, n_steps, o.history_every, o.snapshot_every,
        o.fields, coef_scale, f_host, ω_buf, tips,
    )
end

function history_header(layout::BodyLayout, N, tips=Pair{String,Int}[])
    comps = ("x", "y", "z")[1:N]
    cols = ["step", "t"]
    groups = length(layout.names) > 1 ? [layout.names; "total"] : layout.names
    for g in groups
        append!(cols, ["$(g)_F$c" for c in comps])
        append!(cols, ["$(g)_C$c" for c in comps])
    end
    for (name, _) in tips
        append!(cols, ["$(name)_tip_$c" for c in comps])
    end
    join(cols, ",")
end

# ω, u (and p for IBPM) snapshot datasets, with the level axis kept for every
# formulation: (nx, ny, levels, n_snapshot).
function create_field_datasets!(sg, sol::CNAB{2,T}, fields, n_snap) where {T}
    grid = sol.prob.grid
    st = sol.state
    ibpm = sol.prob.formulation isa IBPM
    levels = ibpm ? 1 : grid.levels

    ω_buf = ibpm ? grid_zeros(get_backend(st.φ), grid, Loc_ω) : nothing
    ω_ref = ibpm ? ω_buf[3] : st.ω[1][3]
    u_ref = ibpm ? st.u_full : st.u[1]

    if "omega" in fields
        d = create_dataset(sg, "omega", T, (size(ω_ref)..., levels, n_snap))
        write_attribute(d, "firstindex", collect(first.(axes(ω_ref))))
    end
    if "u" in fields
        create_dataset(sg, "ux", T, (size(u_ref[1])..., levels, n_snap))
        create_dataset(sg, "uy", T, (size(u_ref[2])..., levels, n_snap))
    end
    if "p" in fields
        create_dataset(sg, "p", T, (size(st.φ)..., 1, n_snap))
    end

    # Physical coordinates of the ω nodes on each level (used by fluid_time_series.jl).
    r = UnitRange.(axes(ω_ref))
    xs = zeros(T, length(r[1]), levels)
    ys = zeros(T, length(r[2]), levels)
    for lev in 1:levels
        X, Y = ibpm ? coord(grid, Loc_ω(3), r) : coord(grid, Loc_ω(3), r, lev)
        xs[:, lev] = X[:, 1]
        ys[:, lev] = Y[:, 1]
    end
    sg["x_coords"] = xs
    sg["y_coords"] = ys
    ω_buf
end

# 3D (FastIBPM): every ω and u component, with its own coordinates per level.
const _COMPONENTS_3D = ("x", "y", "z")

_fields_3d(st, lev, fields) = [
    ("omega" in fields ? [("omega_$c", Loc_ω(i), st.ω[lev][i]) for (i, c) in enumerate(_COMPONENTS_3D)] : [])...,
    ("u" in fields ? [("u$c", Loc_u(i), st.u[lev][i]) for (i, c) in enumerate(_COMPONENTS_3D)] : [])...,
]

function create_field_datasets!(sg, sol::CNAB{3,T}, fields, n_snap) where {T}
    grid = sol.prob.grid
    cg = create_group(sg, "coords")
    for (name, loc, a) in _fields_3d(sol.state, 1, fields)
        d = create_dataset(sg, name, T, (size(a)..., grid.levels, n_snap))
        write_attribute(d, "firstindex", collect(first.(axes(a))))
        r = UnitRange.(axes(a))
        g = create_group(cg, name)
        for (dim, c) in enumerate(_COMPONENTS_3D)
            g[c] = reduce(hcat, [collect(coord(grid, loc, r, lev)[dim]) for lev in 1:grid.levels])
        end
    end
    nothing
end

# IBPM keeps only velocity; recompute ω = ∇×u for output (as cylinder_PV.jl did).
function _ibpm_vorticity!(ω, sol)
    foreach(a -> fill!(a, 0), ω)
    g = sol.prob.grid
    v = grid_view(ω, g, Loc_ω, ExcludeBoundary())
    g isa StretchedGrid ? rot!(v, sol.state.u_full, g) : rot!(v, sol.state.u_full; h=g.h)
    ω
end

function write_step!(out::CaseOutput{N}, sol::CNAB) where {N}
    i = sol.i
    if _selected(i, out.history_every)
        write_history!(out, sol, _slot(i, out.i_first, out.history_every))
    end
    if _selected(i, out.snapshot_every)
        write_snapshot!(out, sol, _slot(i, out.i_first, out.snapshot_every))
    end
    nothing
end

function write_history!(out::CaseOutput{N}, sol, k) where {N}
    f_dev = similar(sol.f_tilde)
    surface_force!(f_dev, sol)
    copyto!(out.f_host, f_dev)

    forces = [sum(view(out.f_host, r); init=zero(SVector{N,Float64})) for r in out.layout.ranges]
    total = sum(forces)
    length(forces) > 1 && push!(forces, total)

    print(out.history, sol.i, ",", sol.t)
    for F in forces
        for c in F
            print(out.history, ",", c)
        end
        for c in F
            print(out.history, ",", c * out.coef_scale)
        end
    end
    tips = [Array(view(sol.points.x, i:i))[1] for (_, i) in out.tips]
    for x in tips, c in x
        print(out.history, ",", c)
    end
    println(out.history)
    flush(out.history)  # so the file can be followed while the run is going

    ag = out.h5["all"]
    ag["t"][k] = sol.t
    ag["Cd"][k] = total[1] * out.coef_scale
    ag["Cl"][k] = total[2] * out.coef_scale
    for (d, x) in zip(tip_dataset_names(out.tips), tips)
        ag[d][:, k] = collect(x)
    end
    ag["force"][:, k] = collect(total)
    total
end

function write_snapshot!(out::CaseOutput{N}, sol::CNAB, k) where {N}
    sg = out.h5["snapshots"]
    sg["t"][k] = sol.t
    sg["body_points"][:, :, k] = reinterpret(reshape, Float64, Array(sol.points.x))

    st = sol.state
    nv(a) = Array(OffsetArrays.no_offset_view(a))
    if N == 3
        for lev in eachindex(st.ω), (name, _, a) in _fields_3d(st, lev, out.fields)
            sg[name][:, :, :, lev, k] = nv(a)
        end
        return nothing
    end
    if sol.prob.formulation isa IBPM
        "omega" in out.fields && (sg["omega"][:, :, 1, k] = nv(_ibpm_vorticity!(out.ω_buf, sol)[3]))
        if "u" in out.fields
            sg["ux"][:, :, 1, k] = nv(st.u_full[1])
            sg["uy"][:, :, 1, k] = nv(st.u_full[2])
        end
        "p" in out.fields && (sg["p"][:, :, 1, k] = nv(st.φ))
    else
        for lev in eachindex(st.ω)
            "omega" in out.fields && (sg["omega"][:, :, lev, k] = nv(st.ω[lev][3]))
            if "u" in out.fields
                sg["ux"][:, :, lev, k] = nv(st.u[lev][1])
                sg["uy"][:, :, lev, k] = nv(st.u[lev][2])
            end
        end
    end
    nothing
end

function write_restart(dir, sol)
    path = joinpath(dir, @sprintf("restart_%07d.bin", sol.i))
    open(io -> save(io, sol), path, "w")
    path
end

function close_output(out::CaseOutput)
    close(out.history)
    close(out.h5)
end
