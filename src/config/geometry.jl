#=
Body geometry generators for case files.

Each generator returns `(x, ds)`: marker positions `Vector{SVector{N,Float64}}` and
quadrature weights `Vector{Float64}` (arc length in 2D, area in 3D). Marker spacing
is `spacing * h`, where `h` is the finest grid step and `spacing` the case-file value
(default 2, as in all the original examples).
=#

function body_geometry(b, h, dim)
    target_ds = b.spacing * h
    g = b.geom
    if b.shape == "cylinder"
        cylinder_points(g.radius, g.center, target_ds)
    elseif b.shape == "plate"
        plate_points(g.start, g.length, g.angle, target_ds)
    elseif b.shape == "naca4"
        naca4_points(g, target_ds)
    elseif b.shape == "points_file"
        points_from_file(g.file, g.closed, dim)
    else
        config_error("[body \"$(b.name)\"] shape \"$(b.shape)\" is not implemented")
    end
end

function cylinder_points(radius, center, target_ds)
    S = 2π * radius
    n = max(3, round(Int, S / target_ds))
    c = SVector{2}(center)
    x = [c + radius * SA[cos(θ), sin(θ)] for θ in range(0, 2π, n + 1)[1:(end-1)]]
    x, fill(S / n, n)
end

# Open curve: `n` points spanning the full length, so spacing is L / (n - 1).
function plate_points(start, len, angle, target_ds)
    n = max(2, 1 + round(Int, len / target_ds))
    x0 = SVector{2}(start)
    t̂ = SA[cos(angle), sin(angle)]
    x = [x0 + s * t̂ for s in range(0, len, n)]
    x, fill(len / (n - 1), n)
end

"""
    naca4_points(g, target_ds)

NACA 4-digit airfoil, re-sampled to uniform arc-length spacing ≈ `target_ds`.
The chord point at fraction `g.pivot_chord` is placed at `g.center`, and the airfoil
is rotated about it by the angle of attack (positive = nose up).
"""
function naca4_points(g, target_ds; n_dense=800)
    m = parse(Int, g.code[1]) / 100
    p = parse(Int, g.code[2]) / 10
    t = parse(Int, g.code[3:4]) / 100

    ξ = range(0.0, 1.0; length=n_dense)
    xc = @. 0.5 * (1 - cos(π * ξ))  # cosine spacing
    c5 = g.closed_te ? 0.1036 : 0.1015
    yt = @. 5t * (0.2969 * sqrt(xc) - 0.1260 * xc - 0.3516 * xc^2 + 0.2843 * xc^3 - c5 * xc^4)

    yc, dyc = zero(xc), zero(xc)
    if m != 0 && p != 0
        for (i, x) in pairs(xc)
            if x < p
                yc[i] = m / p^2 * (2p * x - x^2)
                dyc[i] = 2m / p^2 * (p - x)
            else
                yc[i] = m / (1 - p)^2 * ((1 - 2p) + 2p * x - x^2)
                dyc[i] = 2m / (1 - p)^2 * (p - x)
            end
        end
    end
    θ = atan.(dyc)
    upper = [SA[xc[i] - yt[i] * sin(θ[i]), yc[i] + yt[i] * cos(θ[i])] for i in reverse(eachindex(xc))]
    lower = [SA[xc[i] + yt[i] * sin(θ[i]), yc[i] - yt[i] * cos(θ[i])] for i in eachindex(xc)[2:(end-1)]]
    dense = vcat(upper, lower)  # closed contour TE → upper → LE → lower

    # Scale, rotate about the pivot (nose up = clockwise), and move the pivot to `center`.
    α = -g.angle_of_attack
    R = SA[cos(α) -sin(α); sin(α) cos(α)]
    pivot = SA[g.pivot_chord, 0.0]
    c = SVector{2}(g.center)
    dense = [c + R * (g.chord * (q - pivot)) for q in dense]

    S = sum(norm(dense[mod1(i + 1, end)] - dense[i]) for i in eachindex(dense))
    n = max(40, round(Int, S / target_ds))
    resample_closed(dense, n), fill(S / n, n)
end

# Re-sample a closed polygon to `n` points equally spaced in arc length.
function resample_closed(pts::Vector{SVector{2,T}}, n) where {T}
    N = length(pts)
    s = zeros(T, N + 1)
    for i in 1:N
        s[i+1] = s[i] + norm(pts[mod1(i + 1, N)] - pts[i])
    end
    out = Vector{SVector{2,T}}(undef, n)
    seg = 1
    for (k, sk) in enumerate(range(zero(T), s[end]; length=n + 1)[1:(end-1)])
        while seg < N && sk > s[seg+1]
            seg += 1
        end
        L = s[seg+1] - s[seg]
        α = L == 0 ? zero(T) : (sk - s[seg]) / L
        out[k] = (1 - α) * pts[seg] + α * pts[mod1(seg + 1, N)]
    end
    out
end

"""
    points_from_file(file, closed, dim)

Read markers from a text file: one point per line, columns separated by spaces,
tabs or commas, `#` starts a comment. Columns are `x y [ds]` in 2D and `x y z ds`
in 3D. Without a `ds` column (2D only), each point gets half the distance to its two
neighbours (`closed` decides whether the last point connects back to the first).
"""
function points_from_file(file, closed, dim)
    rows = Vector{Vector{Float64}}()
    for (ln, line) in enumerate(eachline(file))
        line = strip(first(split(line, '#')))
        isempty(line) && continue
        vals = tryparse.(Float64, split(line, r"[\s,;]+"; keepempty=false))
        any(isnothing, vals) && config_error("$file, line $ln: could not parse numbers in \"$line\"")
        push!(rows, vals)
    end
    isempty(rows) && config_error("$file: no points found")
    ncol = length(rows[1])
    all(r -> length(r) == ncol, rows) || config_error("$file: all rows must have the same number of columns")

    if dim == 2 && ncol == 2
        x = [SA[r[1], r[2]] for r in rows]
        x, polyline_weights(x, closed)
    elseif dim == 2 && ncol == 3
        [SA[r[1], r[2]] for r in rows], [r[3] for r in rows]
    elseif dim == 3 && ncol == 4
        [SA[r[1], r[2], r[3]] for r in rows], [r[4] for r in rows]
    else
        expected = dim == 2 ? "2 (x y) or 3 (x y ds)" : "4 (x y z ds)"
        config_error("$file: expected $expected columns for a $(dim)D case, got $ncol")
    end
end

function polyline_weights(x, closed)
    n = length(x)
    n ≥ 2 || config_error("a body needs at least 2 points")
    map(1:n) do i
        prev = i > 1 ? norm(x[i] - x[i-1]) : closed ? norm(x[1] - x[n]) : 0.0
        next = i < n ? norm(x[i+1] - x[i]) : closed ? norm(x[1] - x[n]) : 0.0
        (prev + next) / 2
    end
end
