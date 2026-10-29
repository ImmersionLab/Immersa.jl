#!/usr/bin/env julia
#=
Export the immersed bodies of a case run (solution.h5) to ParaView.

    julia --project=examples examples/post_proc/body_time_series.jl <solution.h5> [--out=DIR]

DIR defaults to <folder of solution.h5>/paraview. Writes
    DIR/body.pvd              time series: open this one in ParaView
    DIR/body/body_NNNN.vtp    markers at each snapshot
(or only DIR/body/body_ref.vtp, the reference shape, when the run saved no snapshots).

Each body is its own cell: a polyline in 2D (closed for cylinders and airfoils, open
for plates and flags), a cloud of points in 3D. Point data:
    force   force of the fluid on each marker (summed over a body it gives the body
            force in history.csv); runs made before this was saved have none
    body    body number (1, 2, ...), to colour or threshold bodies separately
=#

using HDF5
using Printf
using WriteVTK

# =============================================================================
# Command line
# =============================================================================

function parse_args(args)
    isempty(args) && error("usage: body_time_series.jl <solution.h5> [--out=DIR]")
    file = args[1]
    out = joinpath(dirname(abspath(file)), "paraview")
    for a in args[2:end]
        startswith(a, "--out=") ? (out = a[7:end]) : error("unknown option $a")
    end
    (; file, out)
end

# =============================================================================
# Bodies
# =============================================================================

# Marker ranges of each body, and whether each 2D outline is closed. Files without
# body/closed: an outline is closed when its ends are about one marker spacing apart.
function read_bodies(h5)
    bg = h5["body"]
    x_ref = read(bg["points"])
    ranges = haskey(bg, "ranges") ? [r[1]:r[2] for r in eachcol(read(bg["ranges"]))] :
             [1:size(x_ref, 2)]
    names = haskey(bg, "names") ? read(bg["names"]) : ["body$i" for i in eachindex(ranges)]
    closed = if haskey(bg, "closed")
        Bool.(read(bg["closed"]))
    else
        map(ranges) do r
            length(r) < 3 && return false
            gap = sqrt(sum(abs2, x_ref[:, last(r)] - x_ref[:, first(r)]))
            step = sqrt(sum(abs2, x_ref[:, first(r)+1] - x_ref[:, first(r)]))
            gap < 1.5 * step
        end
    end
    (; x_ref, ranges, names, closed, dim=size(x_ref, 1))
end

body_cells(b) =
    b.dim == 3 ? [MeshCell(PolyData.Verts(), [i]) for r in b.ranges for i in r] :
    [MeshCell(PolyData.Lines(), closed ? [r; first(r)] : collect(r))
     for (r, closed) in zip(b.ranges, b.closed)]

as3(x) = size(x, 1) == 3 ? x : vcat(x, zeros(1, size(x, 2)))

function write_body(name, b, cells, x, f)
    vtk = vtk_grid(name, as3(x), cells)
    isnothing(f) || (vtk["force", VTKPointData()] = as3(f))
    id = zeros(Int32, size(x, 2))
    for (i, r) in enumerate(b.ranges)
        id[r] .= i
    end
    vtk["body", VTKPointData()] = id
    vtk
end

# =============================================================================
# Run
# =============================================================================

function main(args)
    (; file, out) = parse_args(args)
    isfile(file) || error("file not found: $file")
    println("Bodies from $file → $out")
    dir = joinpath(out, "body")
    mkpath(dir)

    h5open(file, "r") do h5
        b = read_bodies(h5)
        cells = body_cells(b)
        for (n, r, c) in zip(b.names, b.ranges, b.closed)
            println("  $n: $(length(r)) markers", b.dim == 3 ? "" : c ? ", closed outline" : ", open outline")
        end

        if !haskey(h5, "snapshots") || !haskey(h5["snapshots"], "body_points")
            vtk_save(write_body(joinpath(dir, "body_ref"), b, cells, b.x_ref, nothing))
            println("No snapshots: saved the reference shape, $(joinpath(dir, "body_ref.vtp"))")
            return
        end

        sg = h5["snapshots"]
        time = read(sg["t"])
        has_f = haskey(sg, "force")
        has_f || println("  no snapshots/force in this file (older run): geometry only")

        pvd = paraview_collection(joinpath(out, "body"))
        for k in eachindex(time)
            x = sg["body_points"][:, :, k]
            f = has_f ? sg["force"][:, :, k] : nothing
            pvd[time[k]] = write_body(joinpath(dir, @sprintf("body_%04d", k - 1)), b, cells, x, f)
            (k % 10 == 0 || k == length(time)) && (@printf("\r  snapshot %d/%d", k, length(time)); flush(stdout))
        end
        println()
        vtk_save(pvd)
        println("  saved $(joinpath(out, "body.pvd"))")
    end
end

main(ARGS)
