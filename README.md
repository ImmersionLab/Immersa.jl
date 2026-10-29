# Immersa

*Immersa.jl* is a Julia package that implements immersed boundary methods for the numerical solution of strongly coupled fluid–structure interaction (FSI) problems governed by the incompressible Navier–Stokes equations.

The package includes implementations of:
- The immersed boundary projection method, both in primitive variables ([Taira & Colonius (2007)](https://doi.org/10.1016/j.jcp.2007.03.005)) and in the streamfunction-vorticity formulation ([Colonius & Taira (2008)](https://doi.org/10.1016/j.cma.2007.08.014))
- The strongly coupled FSI formulation of [Goza & Colonius (2017)](https://doi.org/10.1016/j.jcp.2017.02.027)

Prescribed structural deformations are supported in both 2D and 3D flows. Strongly coupled FSI is currently limited to 2D configurations. The solver uses far-field boundary conditions and represents immersed structures as point clouds.

The code uses [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/) for parallel execution on both CPUs and GPUs. It also supports automatic differentiation via [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/) to compute sensitivities of solver outputs.

## Installation
Install the latest version of Julia, then start it and and add the *Immersa* package:

```julia
using Pkg
Pkg.add(url="https://github.com/NUFgroup/Immersa.jl")
using Immersa
```

## Running Examples
Each example is a folder in `examples/` with a `case.toml` case file. The keys are
documented in `examples/case_template.toml`, which is also the starting point for a new case.
- **cylinder_Re200:** static cylinder in a freestream, Re = 200 (vortex shedding).
- **cylinder_Re40 / cylinder_Re40_PV:** static cylinder, Re = 40 (steady wake), streamfunction–vorticity and primitive-variable formulations.
- **moving_cylinder_Re40 / moving_cylinder_Re40_PV:** impulsively started cylinder translating through fluid at rest, Re = 40, both formulations.
- **flag:** flexible flag in a freestream (strongly coupled FSI).
- **naca0012:** rigid airfoil in a freestream.
- **plate:** rigid plate in a freestream.
- **plate_actuation:** rigid plate in a freestream with a prescribed surface velocity (`actuation.jl`).
- **osc_plate:** oscillating rigid plate.
- **tandem_airfoils:** two airfoils in tandem.
- **sphere_Re100 / moving_sphere_Re100:** 3D sphere, Re = 100, static and translating.

## Quick Start
From a clone of the repository, run the cylinder example:

```bash
bin/immersa examples/cylinder_Re200/case.toml
bin/immersa --check examples/cylinder_Re200/case.toml   # validate and print a summary, no run
```
Or from the Julia REPL:
```julia
using Immersa
run_case("examples/cylinder_Re200/case.toml")
```
Results are written to `examples/cylinder_Re200/output/`: the force history in
`history.csv`, and the force history plus field snapshots in `solution.h5`. To view them in
[ParaView](https://www.paraview.org/), convert `solution.h5` with the scripts in
`examples/post_proc` (the files go to `output/paraview/`):
```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'   # once
julia --project=examples examples/post_proc/fluid_time_series.jl examples/cylinder_Re200/output/solution.h5
julia --project=examples examples/post_proc/body_time_series.jl  examples/cylinder_Re200/output/solution.h5
```

## How to Contribute
Contributions are welcome and appreciated! If you would like to improve Immersa.jl, please follow these steps:

### 1. Fork and clone the repository

```bash
git clone https://github.com/NUFgroup/Immersa.jl.git
cd Immersa.jl
```

### 2. Set up the development environment
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
### 3. Create a feature branch
```bash
git checkout -b feature/<short-description>
```
### 4. Make your changes
- Follow the existing code style
- Add or update documentation if needed
- Include tests when applicable

### 5. Run tests
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
### 6. Commit and push
```bash
git add -A
git commit -m "Describe your changes"
git push -u origin feature/<short-description>
```
### 7. Open a pull request
Submit a pull request on GitHub targeting the main branch. Please include a clear description of your changes and their purpose.

### Notes
Use ```--project=.``` to ensure a consistent environment.
For major changes, consider opening an issue first to discuss your approach.

## References
- Colonius, T., & Taira, K. (2008). A fast immersed boundary method using a nullspace approach and multi-domain far-field boundary conditions.
- Goza, A., & Colonius, T. (2017). A strongly-coupled immersed-boundary method for thin elastic structures.
- Taira, K., & Colonius, T. (2007). An immersed boundary method for incompressible flows using primitive variables.
