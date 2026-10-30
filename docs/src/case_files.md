# Case files

A simulation can be set up entirely from a **case file** (`case.toml`), in the style
of SU2 configuration files: you list the parameters, and one command builds and runs
the problem. No Julia code is needed unless you want a custom motion.

## Quick start

```
my_case/
├── case.toml      # all parameters (the only required file)
├── motion.jl      # optional: user-defined prescribed motion
├── body.dat       # optional: imported body markers
└── output/        # written by the run
```

From the shell, at the repository root:

```bash
bin/immersa --check my_case/case.toml   # validate and print the setup summary
bin/immersa my_case/case.toml           # run
```

or from Julia:

```julia
using Immersa
run_case("my_case/case.toml")
```

Start from [`examples/case_template.toml`](https://github.com/NUFgroup/Immersa.jl/blob/main/examples/case_template.toml),
which lists **every key** with its default, or copy one of the example cases in
`examples/*/case.toml`.

## Sections

| Section               | Contents                                                              |
|:----------------------|:----------------------------------------------------------------------|
| `[case]`              | name, backend (`CPU`, `CUDA`, `AMDGPU`)                               |
| `[flow]`              | `Re`, `freestream` (its length sets 2D/3D), smooth start, user inflow |
| `[grid]`              | uniform multi-domain grid, or stretched grid (IBPM)                   |
| `[[body]]`            | one block per body: shape, marker spacing, optional motion/structure  |
| `[solver]`            | formulation, `dt`, `t_final`, delta function, restart                 |
| `[initial_condition]` | optional vorticity blob to trigger shedding                           |
| `[reference]`         | velocity, length (2D) or area (3D) for force coefficients             |
| `[output]`            | history/snapshot frequency, fields, checkpoints                       |

All angles in a case file are in **degrees**; relative paths are relative to the case
folder. Unknown keys are an error (with a suggestion for likely typos), so a misspelt
parameter can never be silently ignored.

## User-defined motion

With `type = "user"` in `[body.motion]`, the Julia file named by `file` must define
exactly one of:

```julia
# Rigid motion relative to the reference shape (angle in radians):
rigid_motion(t, params) = (; center = [0.0, 0.1sin(t)], velocity = [0.0, 0.1cos(t)],
                             angle = 0.0, angular_velocity = 0.0)

# General prescribed motion / deformation: overwrite positions and velocities.
function motion!(x, u, x_ref, t, params) ... end

# Fixed body with an actuated surface: set the marker velocities.
function surface_velocity!(u, x_ref, t, params) ... end
```

`params` is the `[body.motion.parameters]` table (it can be omitted from the
signature). `StaticArrays` and `LinearAlgebra` are available in the file.

## Output

| File                         | Contents                                                       |
|:-----------------------------|:---------------------------------------------------------------|
| `output/history.csv`         | step, time, force and coefficient per body (+ total)           |
| `output/solution.h5`         | force history, field snapshots, and the body markers and the force on each marker at every snapshot |
| `output/restart_<step>.bin`  | checkpoints, for `[solver] restart_from`                       |
| `output/case.toml`           | copy of the case file used                                     |
| `output/paraview/`           | ParaView files, written by the `examples/post_proc` scripts (below) |

### ParaView

The scripts in `examples/post_proc` convert `solution.h5` to ParaView files, by default
in `paraview/` next to it:

```bash
julia --project=examples examples/post_proc/fluid_time_series.jl examples/cylinder_Re200/output/solution.h5
julia --project=examples examples/post_proc/body_time_series.jl  examples/cylinder_Re200/output/solution.h5
```

Open `fluid_level1.pvd` and `body.pvd` in ParaView.

- **Fluid:** one rectilinear grid per snapshot. In 2D it holds vorticity on the nodes
  and velocity (and `p` for IBPM) at the cell centres; in 3D velocity and vorticity at
  the cell centres. Only the finest level is exported by default; use `--levels=1,2`
  or `--levels=all` for the others (level 1 = finest).
- **Body:** the markers at each snapshot, with the force of the fluid on each marker
  and the body number. In 2D each body is a polyline (open for plates); in 3D a
  cloud of points.
- `--out=DIR` writes somewhere else. The first time, set up the environment with
  `julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`.

## 3D cases

A three-entry `freestream` makes the case 3D. Currently supported: FastIBPM, rigid
bodies (`shape = "sphere"` or `points_file`), static or translating/oscillating
(no 3D rotation yet). See `examples/sphere_Re100` and `examples/moving_sphere_Re100`.

- Marker spacing defaults to `spacing = 1.0` (in units of `h`); each marker carries
  an area weight of about `(spacing·h)²`.
- Force coefficients use `[reference] area` (e.g. the frontal area πr² of a sphere).
- A static body's coupling matrix is assembled once at start-up, with one 3D Poisson
  solve per marker component: expect minutes to tens of minutes for ~10³ markers.
- `solution.h5` stores every component on every level, each on its own staggered
  points: `snapshots/omega_x|omega_y|omega_z|ux|uy|uz` with shape
  `(n1, n2, n3, levels, n_snapshot)`, and their coordinates in
  `snapshots/coords/<field>/x|y|z` with shape `(n_d, levels)`.

## API

```@docs
run_case
load_case
build_case
build_solver
CaseConfigError
```
