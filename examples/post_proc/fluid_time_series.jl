#!/usr/bin/env julia
#=
Export the field snapshots of a case run (solution.h5) to ParaView.

    julia --project=examples examples/post_proc/fluid_time_series.jl <solution.h5> [options]

Options
    --levels=1        grid levels to export: 1 (finest, the default), e.g. 1,2, or all
    --out=DIR         output folder (default: <folder of solution.h5>/paraview)

Writes, for each level L:
    DIR/fluid_level<L>.pvd             time series: open this one in ParaView
    DIR/fluid_level<L>/fluid_NNNN.vtr  one rectilinear grid per snapshot

Fields (whichever were saved, see [output] fields in the case file), on the grid of
cell corners (nodes):
    2D  vorticity (point data, it lives on the nodes), velocity and p (cell data;
        the staggered u, v are averaged to the cell centres)
    3D  velocity and vorticity (cell data, every component averaged to the centres)
Works for FastIBPM (multi-level), IBPM (one level, uniform or stretched) and 3D runs.
=#

using HDF5
using Printf
using WriteVTK

# =============================================================================
# Command line
# =============================================================================

function parse_args(args)
    isempty(args) && error("usage: fluid_time_series.jl <solution.h5> [--levels=1|1,2|all] [--out=DIR]")
    file = args[1]
    levels = "1"
    out = joinpath(dirname(abspath(file)), "paraview")
    for a in args[2:end]
        if startswith(a, "--levels=")
            levels = a[10:end]
        elseif startswith(a, "--out=")
            out = a[7:end]
        else
            error("unknown option $a")
        end
    end
    (; file, levels, out)
end

function select_levels(spec, n_levels)
    spec == "all" && return collect(1:n_levels)
    levs = parse.(Int, split(spec, ","))
    for l in levs
        1 <= l <= n_levels || error("level $l does not exist: the file has levels 1:$n_levels (1 = finest)")
    end
    levs
end

# =============================================================================
# Staggered data → cell centres
# =============================================================================

# Average `a` along every dimension where it has one more point than there are cells
# (i.e. where it sits on the nodes), so the result has size `ncells`.
function to_cells(a::AbstractArray, ncells)
    for d in 1:ndims(a)
        n = size(a, d)
        if n == ncells[d] + 1
            lo = selectdim(a, d, 1:(n - 1))
            hi = selectdim(a, d, 2:n)
            a = (lo .+ hi) ./ 2
        elseif n != ncells[d]
            error("field size $(size(a)) does not fit a grid of $ncells cells")
        end
    end
    a
end

# =============================================================================
# 2D
# =============================================================================

function export_2d(h5, levels, out, time)
    sg = h5["snapshots"]
    has(k) = haskey(sg, k)
    has("x_coords") || error("snapshots/x_coords missing: not a case-file solution.h5?")
    xs = read(sg["x_coords"]); ys = read(sg["y_coords"])
    n_levels = size(xs, 2)

    for lev in select_levels(levels, n_levels)
        x = xs[:, lev]; y = ys[:, lev]          # node (cell corner) coordinates
        ncells = (length(x) - 1, length(y) - 1)
        dir = joinpath(out, "fluid_level$lev")
        mkpath(dir)
        pvd = paraview_collection(joinpath(out, "fluid_level$lev"))
        println("Level $lev: x ∈ [$(x[1]), $(x[end])], y ∈ [$(y[1]), $(y[end])], ",
                "$(ncells[1]) × $(ncells[2]) cells")

        for k in eachindex(time)
            name = joinpath(dir, @sprintf("fluid_%04d", k - 1))
            vtk = vtk_grid(name, x, y)
            if has("omega")
                vtk["vorticity", VTKPointData()] = sg["omega"][:, :, lev, k]
            end
            if has("ux")
                u = to_cells(sg["ux"][:, :, lev, k], ncells)
                v = to_cells(sg["uy"][:, :, lev, k], ncells)
                vtk["velocity", VTKCellData()] = (u, v, zero(u))
            end
            if has("p") && lev <= size(sg["p"], 3)
                vtk["p", VTKCellData()] = to_cells(sg["p"][:, :, lev, k], ncells)
            end
            pvd[time[k]] = vtk
            progress(k, length(time))
        end
        vtk_save(pvd)
        println("  saved $(joinpath(out, "fluid_level$lev.pvd"))")
    end
end

# =============================================================================
# 3D
# =============================================================================

const COMPONENTS = ("x", "y", "z")

function export_3d(h5, levels, out, time)
    sg = h5["snapshots"]
    cg = sg["coords"]
    has(k) = haskey(sg, k)
    has_u = has("ux"); has_ω = has("omega_x")
    # Node coordinates along d: u_d sits on the faces normal to d, i.e. on the nodes
    # along d (fall back to the ω components, which sit on the nodes across them).
    node_source(d) = has_u ? ("u" * COMPONENTS[d], d) :
                     ("omega_" * COMPONENTS[d % 3 + 1], d)
    n_levels = size(cg[node_source(1)[1]]["x"], 2)

    for lev in select_levels(levels, n_levels)
        xyz = [read(cg[node_source(d)[1]][COMPONENTS[d]])[:, lev] for d in 1:3]
        ncells = Tuple(length.(xyz) .- 1)
        dir = joinpath(out, "fluid_level$lev")
        mkpath(dir)
        pvd = paraview_collection(joinpath(out, "fluid_level$lev"))
        println("Level $lev: ", join(["$c ∈ [$(r[1]), $(r[end])]" for (c, r) in zip(COMPONENTS, xyz)], ", "),
                ", $(join(ncells, " × ")) cells")

        for k in eachindex(time)
            name = joinpath(dir, @sprintf("fluid_%04d", k - 1))
            vtk = vtk_grid(name, xyz...)
            if has_u
                vtk["velocity", VTKCellData()] =
                    Tuple(to_cells(sg["u$c"][:, :, :, lev, k], ncells) for c in COMPONENTS)
            end
            if has_ω
                vtk["vorticity", VTKCellData()] =
                    Tuple(to_cells(sg["omega_$c"][:, :, :, lev, k], ncells) for c in COMPONENTS)
            end
            pvd[time[k]] = vtk
            progress(k, length(time))
        end
        vtk_save(pvd)
        println("  saved $(joinpath(out, "fluid_level$lev.pvd"))")
    end
end

progress(k, n) = (k % 10 == 0 || k == n) && (@printf("\r  snapshot %d/%d", k, n); k == n && println(); flush(stdout))

# =============================================================================
# Run
# =============================================================================

function main(args)
    (; file, levels, out) = parse_args(args)
    isfile(file) || error("file not found: $file")
    println("Fluid snapshots from $file → $out")
    h5open(file, "r") do h5
        haskey(h5, "snapshots") || error("no snapshots in $file (snapshot_every = 0?)")
        sg = h5["snapshots"]
        time = read(sg["t"])
        println("$(length(time)) snapshots, t ∈ [$(time[1]), $(time[end])]")
        if haskey(sg, "coords")
            export_3d(h5, levels, out, time)
        else
            export_2d(h5, levels, out, time)
        end
    end
end

main(ARGS)
