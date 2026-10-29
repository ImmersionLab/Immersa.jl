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
    # Deformable bodies: the fluid–structure iteration tolerance.
    flag = BASE * """
    [[body]]
    model  = "deformable"
    shape  = "plate"
    length = 1.0
    start  = [-0.5, 0.0]
      [body.structure]
      mass                  = 0.1
      bending_stiffness     = 0.01
      extensional_stiffness = 100.0
    """
    fsi_tol(text) = mktempdir() do dir
        build_solver(build_case(write_case(dir, text))).coupler.tol
    end
    @test fsi_tol(flag) == 1e-5
    @test fsi_tol(replace(flag, "dt      = 0.01" => "dt = 0.01\ncoupling_tol = 1e-7")) == 1e-7
end

const BASE_3D = """
[case]
name = "tiny3d"

[flow]
Re         = 50.0
freestream = [1.0, 0.0, 0.0]

[grid]
h      = 0.1
limits = [[-1.2, 2.0], [-1.2, 1.2], [-1.2, 1.2]]
levels = 2

[[body]]
name    = "sphere"
shape   = "sphere"
radius  = 0.4
spacing = 2.0

[solver]
dt      = 0.02
t_final = 0.06

[reference]
area = 0.5026548245743669

[output]
snapshot_every = 2
log_every      = 0
"""

function test_sphere_geometry()
    x, w = Immersa.sphere_points(0.5, [1.0, 2.0, 3.0], 0.05)
    @test sum(w) ≈ 4π * 0.25
    @test all(p -> norm(p - SA[1.0, 2.0, 3.0]) ≈ 0.5, x)
    nn = [minimum(norm(x[i] - x[j]) for j in eachindex(x) if j != i) for i in eachindex(x)]
    @test 0.8 * 0.05 < minimum(nn) && maximum(nn) < 1.1 * 0.05
end

function test_3d_static()
    mktempdir() do dir
        sol = quiet_run(write_case(dir, BASE_3D))
        @test sol.i == 4
        @test startswith(readlines(joinpath(dir, "output", "history.csv"))[1],
            "step,t,sphere_Fx,sphere_Fy,sphere_Fz,sphere_Cx,sphere_Cy,sphere_Cz")
        F = surface_force_sum(sol)
        @test F[1] > 0                                   # drag along the flow
        @test abs(F[2]) < 1e-2 * F[1] && abs(F[3]) < 1e-2 * F[1]  # ~0 by symmetry
        h5open(joinpath(dir, "output", "solution.h5")) do f
            for name in ("omega_x", "omega_y", "omega_z", "ux", "uy", "uz")
                d = read(f["snapshots/$name"])
                @test ndims(d) == 5 && size(d)[4:5] == (2, 2)   # levels, snapshots (steps 1, 3)
                for (k, c) in enumerate(("x", "y", "z"))
                    @test size(read(f["snapshots/coords/$name/$c"])) == (size(d, k), 2)
                end
            end
            @test read(f["all/Cd"])[end] ≈ F[1] / (0.5 * 0.5026548245743669)
        end
    end
end

function test_3d_moving()
    text = replace(BASE_3D, "freestream = [1.0, 0.0, 0.0]" => "freestream = [0.0, 0.0, 0.0]",
        "spacing = 2.0" => """
        spacing = 2.0
          [body.motion]
          type     = "translate"
          velocity = [-1.0, 0.0, 0.0]
        """)
    mktempdir() do dir
        sol = quiet_run(write_case(dir, text))
        x_ref = build_case(joinpath(dir, "case.toml")).layout.x_ref
        @test sol.points.x ≈ [x + SA[-sol.t, 0.0, 0.0] for x in x_ref]
        @test all(==(SA[-1.0, 0.0, 0.0]), sol.points.u)
        @test surface_force_sum(sol)[1] > 0                # body moves in -x, drag acts in +x
    end
end

function test_tip_tracking()
    text = BASE * """
    [[body]]
    name   = "flag"
    model  = "deformable"
    shape  = "plate"
    length = 1.0
    start  = [-0.5, 0.0]
    track_tip = true

      [body.structure]
      mass                  = 0.1
      bending_stiffness     = 0.01
      extensional_stiffness = 100.0
      initial_angle         = 2.0
    """
    mktempdir() do dir
        sol = quiet_run(write_case(dir, text))
        lines = readlines(joinpath(dir, "output", "history.csv"))
        @test endswith(lines[1], ",flag_tip_x,flag_tip_y")
        last_row = parse.(Float64, split(lines[end], ","))
        @test last_row[end-1:end] ≈ collect(sol.points.x[end])   # trailing edge = last marker
        h5open(joinpath(dir, "output", "solution.h5")) do f
            x_tip = read(f["all/x_tip"])
            @test size(x_tip) == (2, 6)
            @test x_tip[:, end] ≈ collect(sol.points.x[end])
        end
    end
    # Opt-in only: without the key there are no tip columns...
    mktempdir() do dir
        quiet_run(write_case(dir, replace(text, "track_tip = true\n" => "")))
        @test !occursin("tip", readlines(joinpath(dir, "output", "history.csv"))[1])
    end
    # ...and it is rejected for shapes without a free end.
    @test occursin("only available for shape = \"plate\"",
        load_error(BASE * CYLINDER * "track_tip = true\n"))
end

function run_all()
    @testset "validation" test_validation()
    @testset "static run" test_static_run()
    @testset "moving + two bodies" test_moving_two_bodies()
    @testset "user motion file" test_user_motion()
    @testset "restart" test_restart()
    @testset "coupling tolerance" test_coupling_tol()
    @testset "sphere geometry" test_sphere_geometry()
    @testset "3D static sphere" test_3d_static()
    @testset "3D moving sphere" test_3d_moving()
    @testset "trailing-edge tracking" test_tip_tracking()
end

end # module
