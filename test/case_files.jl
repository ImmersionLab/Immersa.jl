# Tests for case files (src/config/): validation messages, and tiny end-to-end runs
# (static, built-in motion, multiple bodies, user motion file, restart).

module CaseFileTests

using Immersa
using HDF5
using StaticArrays
using LinearAlgebra
using Test

const BASE = """
[case]
name = "tiny"

[flow]
Re         = 100.0
freestream = [1.0, 0.0]

[grid]
h      = 0.1
limits = [[-2.0, 2.0], [-2.0, 2.0]]
levels = 2

[solver]
dt      = 0.01
t_final = 0.05

[output]
snapshot_every = 2
log_every      = 0
"""

const CYLINDER = """
[[body]]
name   = "cyl"
shape  = "cylinder"
radius = 0.5
"""

function write_case(dir, text; files=Dict{String,String}())
    mkpath(dir)
    write(joinpath(dir, "case.toml"), text)
    for (name, content) in files
        write(joinpath(dir, name), content)
    end
    joinpath(dir, "case.toml")
end

quiet_run(path) = redirect_stdout(() -> run_case(path), devnull)

# The error message thrown when loading `text`.
function load_error(text)
    mktempdir() do dir
        try
            load_case(write_case(dir, text))
        catch e
            e isa CaseConfigError || rethrow()
            return e.msg
        end
        return ""
    end
end

function test_validation()
    @test occursin("found `raduis` — did you mean `radius`", load_error(BASE * replace(CYLINDER, "radius" => "raduis")))
    @test occursin("unknown key `spacng` (did you mean `spacing`?)", load_error(BASE * CYLINDER * "spacng = 2.0\n"))
    @test occursin("missing required key `Re`", load_error(replace(BASE, "Re         = 100.0" => "") * CYLINDER))
    @test occursin("unknown section [solvr]", load_error(replace(BASE, "[solver]" => "[solvr]") * CYLINDER))
    @test occursin("`dt` must be a number", load_error(replace(BASE, "0.01" => "\"x\"") * CYLINDER))
    @test occursin("allowed values", load_error(BASE * replace(CYLINDER, "\"cylinder\"" => "\"cube\"")))
    @test occursin("must have 2 entries", load_error(BASE * CYLINDER * "center = [0.0, 0.0, 0.0]\n"))
    @test occursin("at least one [[body]]", load_error(BASE))
    @test occursin("(double brackets)", load_error(BASE * replace(CYLINDER, "[[body]]" => "[body]")))
    @test occursin("requires [solver] formulation = \"IBPM\"",
        load_error(replace(BASE, "h      = 0.1" => "type = \"stretched\"\nh = 0.1") * CYLINDER))
end

function test_static_run()
    mktempdir() do dir
        sol = quiet_run(write_case(dir, BASE * CYLINDER))
        @test sol.i == 6 && sol.t ≈ 0.05
        lines = readlines(joinpath(dir, "output", "history.csv"))
        @test lines[1] == "step,t,cyl_Fx,cyl_Fy,cyl_Cx,cyl_Cy"
        @test length(lines) == 1 + 6
        h5open(joinpath(dir, "output", "solution.h5")) do f
            @test length(read(f["all/Cd"])) == 6
            @test size(read(f["snapshots/omega"]))[end-1:end] == (2, 3)  # levels, snapshots 1,3,5
            @test read(f["all/Cd"])[end] ≈ 2 * surface_force_sum(sol)[1]
        end
        @test isfile(joinpath(dir, "output", "case.toml"))
    end
end

function test_moving_two_bodies()
    text = BASE * CYLINDER * """
    [[body]]
    name   = "plate"
    shape  = "plate"
    length = 0.6
    center = [1.2, 0.0]
    angle  = 90.0

      [body.motion]
      type      = "oscillate"
      amplitude = [0.0, 0.2]
      frequency = 1.0
    """
    mktempdir() do dir
        sol = quiet_run(write_case(dir, text))
        setup = build_case(joinpath(dir, "case.toml"))
        r_cyl, r_plate = setup.layout.ranges
        x_ref = setup.layout.x_ref
        t = sol.t
        # Static cylinder untouched; plate displaced by A sin(2πft).
        @test sol.points.x[r_cyl] == x_ref[r_cyl]
        @test sol.points.x[r_plate] ≈ [x + SA[0.0, 0.2 * sin(2π * t)] for x in x_ref[r_plate]]
        @test all(u -> u ≈ SA[0.0, 0.2 * 2π * cos(2π * t)], sol.points.u[r_plate])
        @test startswith(readlines(joinpath(dir, "output", "history.csv"))[1],
            "step,t,cyl_Fx,cyl_Fy,cyl_Cx,cyl_Cy,plate_Fx")
    end
end

function test_user_motion()
    motion = """
    # Rotation about the pivot at a constant rate (params.rate, rad/time).
    rigid_motion(t, params) = (; angle = params.rate * t, angular_velocity = params.rate)
    """
    text = BASE * """
    [[body]]
    shape  = "plate"
    length = 1.0

      [body.motion]
      type  = "user"
      file  = "motion.jl"
      pivot = [0.0, 0.0]

        [body.motion.parameters]
        rate = 2.0
    """
    mktempdir() do dir
        sol = quiet_run(write_case(dir, text; files=Dict("motion.jl" => motion)))
        x_ref = build_case(joinpath(dir, "case.toml")).layout.x_ref
        θ = 2.0 * sol.t
        R = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]
        @test sol.points.x ≈ [R * x for x in x_ref]
        @test sol.points.u ≈ [2.0 * SA[-(R*x)[2], (R*x)[1]] for x in x_ref]
    end

    # A user file must define exactly one motion function.
    mktempdir() do dir
        path = write_case(dir, text; files=Dict("motion.jl" => "f(t) = t"))
        @test_throws CaseConfigError build_case(path)
    end
end

function test_restart()
    mktempdir() do dir
        full = quiet_run(write_case(joinpath(dir, "full"), BASE * CYLINDER))

        first_half = replace(BASE, "t_final = 0.05" => "t_final = 0.02") *
                     "restart_every = 3\n" * CYLINDER
        quiet_run(write_case(joinpath(dir, "a"), first_half))
        ckpt = joinpath(dir, "a", "output", "restart_0000003.bin")
        @test isfile(ckpt)

        second = replace(BASE, "dt      = 0.01" => "dt = 0.01\nrestart_from = \"$ckpt\"") * CYLINDER
        resumed = quiet_run(write_case(joinpath(dir, "b"), second))
        @test resumed.i == full.i
        @test surface_force_sum(resumed) ≈ surface_force_sum(full)
    end
end

function test_coupling_tol()
    moving = BASE * replace(CYLINDER, "radius = 0.5" => """
    radius = 0.5
      [body.motion]
      type     = "translate"
      velocity = [0.1, 0.0]
    """)
    abstol(text) = mktempdir() do dir
        build_solver(build_case(write_case(dir, text))).coupler.Binv.abstol
    end
    @test abstol(moving) == Immersa.FASTIBPM_COUPLING_ABSTOL == 1e-7
    @test abstol(replace(moving, "dt      = 0.01" => "dt = 0.01\ncoupling_tol = 1e-9")) == 1e-9
    ibpm = replace(BASE, "dt      = 0.01" => "dt = 0.01\nformulation = \"IBPM\"", "levels = 2" => "levels = 1")
    @test abstol(ibpm * CYLINDER) == Immersa.CNAB_Binv_Iterative{Float64}().abstol  # IBPM default unchanged
end

function run_all()
    @testset "validation" test_validation()
    @testset "static run" test_static_run()
    @testset "moving + two bodies" test_moving_two_bodies()
    @testset "user motion file" test_user_motion()
    @testset "restart" test_restart()
    @testset "coupling tolerance" test_coupling_tol()
end

end # module
