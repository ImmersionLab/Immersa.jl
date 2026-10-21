# GroupedPrescribedBody: each sub-body must own its slice of the marker arrays, and
# a group must reproduce the equivalent single body exactly.

module GroupedBodyTests

using Immersa
using Immersa: GroupedPrescribedBody, body_ranges, CNAB_Binv_Iterative
using StaticArrays
using LinearAlgebra
using Random
using Test

const H = 0.1
const GRID = Grid(; h=H, n=SA[40, 40], x0=SA[-2.0, -2.0], levels=2)

circle(c, r, n) = [c + r * SA[cos(θ), sin(θ)] for θ in range(0, 2π, n + 1)[1:(end-1)]]

const XA = circle(SA[-0.8, 0.0], 0.4, 13)
const XB = circle(SA[0.8, 0.0], 0.4, 13)
const DS = fill(2π * 0.4 / 13, 13)

# Body B heaves: y = 0.1 sin(2t).
heave!(x, u, x_ref, t) =
    for j in eachindex(x)
        x[j] = x_ref[j] + SA[0.0, 0.1 * sin(2t)]
        u[j] = SA[0.0, 0.2 * cos(2t)]
    end

function forces(body; nstep=10)
    prob = IBProblem(GRID, body, 100.0, UniformFlow(t -> SA[1.0, 0.0]))
    sol = CNAB(prob; dt=0.01)
    Random.seed!(1)  # BiCGStab(ℓ) draws a random shadow residual
    F = [(step!(sol); surface_force_sum(sol)) for _ in 1:nstep]
    F, sol
end

function test_ranges()
    g = GroupedPrescribedBody((StaticBody(XA, DS), StaticBody(XB[1:5], DS[1:5])))
    @test body_ranges(g) == (1:13, 14:18)
    @test Immersa.point_count(g) == 18
end

function test_static_group()
    F1, sol1 = forces(StaticBody([XA; XB], [DS; DS]))
    F2, sol2 = forces(GroupedPrescribedBody((StaticBody(XA, DS), StaticBody(XB, DS))))
    @test !(sol2.coupler.Binv isa CNAB_Binv_Iterative)   # precomputed static path
    @test sol2.points.x == [XA; XB]
    @test maximum(norm, F1 .- F2) < 1e-10
end

function test_mixed_group()
    xr = [XA; XB]
    merged!(x, u, i, t) = begin
        u[1:13] .= Ref(zero(SVector{2,Float64}))
        heave!(view(x, 14:26), view(u, 14:26), XB, t)
    end
    F1, sol1 = forces(MovingBody(xr, [DS; DS], merged!))
    grouped = GroupedPrescribedBody((
        StaticBody(XA, DS), MovingBody(XB, DS, (x, u, i, t) -> heave!(x, u, XB, t))
    ))
    F2, sol2 = forces(grouped)
    @test sol2.coupler.Binv isa CNAB_Binv_Iterative     # moving → per-step solve
    @test sol2.points.x[1:13] == XA                      # static part untouched
    @test sol2.points.x[14:26] ≈ [x + SA[0.0, 0.1 * sin(2 * sol2.t)] for x in XB]
    @test sol2.points.x == sol1.points.x
    @test maximum(norm, F1 .- F2) < 1e-10
end

function run_all()
    @testset "ranges" test_ranges()
    @testset "static group = single static body" test_static_group()
    @testset "static + moving group = single moving body" test_mixed_group()
end

end # module
