#=
Case-file parsing and validation.

`load_case(path)` reads a `case.toml`, checks every key (type, allowed values,
required/optional, unknown keys) and returns a fully resolved `CaseConfig`: a plain
tree of NamedTuples with every default filled in. The builders (`builders.jl`) and
the driver (`driver.jl`) only ever see this resolved tree, never the raw TOML.

All lengths/times are in the solver's non-dimensional units. All angles in the case
file are in DEGREES; they are converted to radians here.
=#

"""
    CaseConfigError(msg)

Error thrown for any problem in a case file (missing/unknown key, wrong type, value
out of range). The message always names the offending section and key.
"""
struct CaseConfigError <: Exception
    msg::String
end
Base.showerror(io::IO, e::CaseConfigError) = print(io, "CaseConfigError: ", e.msg)

config_error(msg) = throw(CaseConfigError(msg))

# ---------------------------------------------------------------------------
# Section: a TOML table that remembers which keys were read, so that unknown
# (e.g. misspelled) keys can be reported instead of silently ignored.
# ---------------------------------------------------------------------------

struct Section
    name::String
    dict::Dict{String,Any}
    used::Set{String}
end
Section(name, dict) = Section(name, dict, Set{String}())

struct _Required end
const REQUIRED = _Required()

Base.haskey(s::Section, key) = haskey(s.dict, key)

function _raw(s::Section, key, default)
    push!(s.used, key)
    haskey(s.dict, key) && return s.dict[key]
    if default isa _Required
        typo = _closest(key, setdiff(keys(s.dict), s.used))
        config_error("[$(s.name)] missing required key `$key`" *
                     (isnothing(typo) ? "" : " (found `$typo` — did you mean `$key`?)"))
    end
    default
end

function subsection(s::Section, key; required=false)
    push!(s.used, key)
    if !haskey(s.dict, key)
        required && config_error("[$(s.name)] missing required section [$(s.name).$key]")
        return nothing
    end
    d = s.dict[key]
    d isa AbstractDict || config_error("[$(s.name)] `$key` must be a table [$(s.name).$key]")
    Section("$(s.name).$key", d)
end

function check_unknown_keys(s::Section)
    unknown = setdiff(keys(s.dict), s.used)
    isempty(unknown) && return nothing
    k = first(sort!(collect(unknown)))
    hint = _closest(k, s.used)
    config_error(
        "[$(s.name)] unknown key `$k`" * (isnothing(hint) ? "" : " (did you mean `$hint`?)")
    )
end

# Suggest the closest known key for a typo (Levenshtein distance ≤ 3).
function _closest(k, candidates)
    best, bestd = nothing, 4
    for c in candidates
        d = _levenshtein(lowercase(k), lowercase(c))
        d < bestd && ((best, bestd) = (c, d))
    end
    best
end

function _levenshtein(a, b)
    a, b = collect(a), collect(b)
    d = collect(0:length(b))
    for i in eachindex(a)
        prev, d[1] = d[1], i
        for j in eachindex(b)
            cur = d[j+1]
            d[j+1] = min(d[j+1] + 1, d[j] + 1, prev + (a[i] != b[j]))
            prev = cur
        end
    end
    d[end]
end

# ---------------------------------------------------------------------------
# Typed getters
# ---------------------------------------------------------------------------

function get_float(s::Section, key, default=REQUIRED; min=-Inf, max=Inf, strict_min=false)
    v = _raw(s, key, default)
    v isa Real || config_error("[$(s.name)] `$key` must be a number, got $(repr(v))")
    v = Float64(v)
    (strict_min ? v <= min : v < min) &&
        config_error("[$(s.name)] `$key` must be $(strict_min ? ">" : "≥") $min, got $v")
    v > max && config_error("[$(s.name)] `$key` must be ≤ $max, got $v")
    v
end

function get_int(s::Section, key, default=REQUIRED; min=typemin(Int))
    v = _raw(s, key, default)
    (v isa Integer || (v isa AbstractFloat && isinteger(v))) ||
        config_error("[$(s.name)] `$key` must be an integer, got $(repr(v))")
    v = Int(v)
    v < min && config_error("[$(s.name)] `$key` must be ≥ $min, got $v")
    v
end

function get_bool(s::Section, key, default=REQUIRED)
    v = _raw(s, key, default)
    v isa Bool || config_error("[$(s.name)] `$key` must be true or false, got $(repr(v))")
    v
end

function get_string(s::Section, key, default=REQUIRED; allowed=nothing)
    v = _raw(s, key, default)
    v isa AbstractString || config_error("[$(s.name)] `$key` must be a string, got $(repr(v))")
    if !isnothing(allowed) && !(v in allowed)
        config_error(
            "[$(s.name)] `$key` = \"$v\" is not valid; allowed values: " *
            join(("\"$a\"" for a in allowed), ", "),
        )
    end
    String(v)
end

function get_vector(s::Section, key, default=REQUIRED; len=nothing)
    v = _raw(s, key, default)
    (v isa AbstractVector && all(x -> x isa Real, v)) ||
        config_error("[$(s.name)] `$key` must be a list of numbers, got $(repr(v))")
    !isnothing(len) && length(v) != len &&
        config_error("[$(s.name)] `$key` must have $len entries (problem is $(len)D), got $(length(v))")
    Float64.(v)
end

# A list of [lo, hi] pairs, one per dimension, e.g. limits = [[-1, 3], [-2, 2]].
function get_limits(s::Section, key; len=nothing)
    v = _raw(s, key, REQUIRED)
    ok = v isa AbstractVector && all(p -> p isa AbstractVector && length(p) == 2 &&
                                          all(x -> x isa Real, p), v)
    ok || config_error("[$(s.name)] `$key` must be a list of [min, max] pairs, e.g. [[-1.0, 3.0], [-2.0, 2.0]]")
    !isnothing(len) && length(v) != len &&
        config_error("[$(s.name)] `$key` must have $len [min, max] pairs (problem is $(len)D), got $(length(v))")
    for (d, p) in enumerate(v)
        p[1] < p[2] || config_error("[$(s.name)] `$key`: entry $d has min ≥ max ($(p[1]) ≥ $(p[2]))")
    end
    [Float64(p[k]) for p in v, k in 1:2]  # N×2 matrix
end

function get_string_list(s::Section, key, default=REQUIRED; allowed=nothing)
    v = _raw(s, key, default)
    (v isa AbstractVector && all(x -> x isa AbstractString, v)) ||
        config_error("[$(s.name)] `$key` must be a list of strings, got $(repr(v))")
    if !isnothing(allowed)
        for x in v
            x in allowed || config_error(
                "[$(s.name)] `$key` contains \"$x\"; allowed values: " *
                join(("\"$a\"" for a in allowed), ", "),
            )
        end
    end
    String.(v)
end

# A path in the case file is relative to the case folder.
function get_path(s::Section, key, casedir, default=REQUIRED)
    v = get_string(s, key, default)
    p = isabspath(v) ? v : joinpath(casedir, v)
    isfile(p) || config_error("[$(s.name)] `$key`: file not found: $p")
    p
end

# ---------------------------------------------------------------------------
# Resolved configuration
# ---------------------------------------------------------------------------

"""
    CaseConfig

Fully validated case description produced by [`load_case`](@ref). Fields are
NamedTuples with every default filled in:
`case, flow, grid, bodies, solver, initial_condition, reference, output`, plus
`dir` (the case folder) and `file` (the case-file path).
"""
struct CaseConfig
    file::String
    dir::String
    dim::Int
    case::NamedTuple
    flow::NamedTuple
    grid::NamedTuple
    bodies::Vector{NamedTuple}
    solver::NamedTuple
    initial_condition::NamedTuple
    reference::NamedTuple
    output::NamedTuple
end

const TOP_LEVEL_SECTIONS =
    ("case", "flow", "grid", "body", "solver", "initial_condition", "reference", "output")

"""
    load_case(path) -> CaseConfig

Parse and validate the case file at `path` (a `.toml` file, or a folder containing
`case.toml`). Throws a [`CaseConfigError`](@ref) describing the first problem found.
"""
function load_case(path::AbstractString)
    file = isdir(path) ? joinpath(path, "case.toml") : path
    isfile(file) || config_error("case file not found: $file")
    file = abspath(file)
    dir = dirname(file)

    raw = try
        TOML.parsefile(file)
    catch e
        e isa TOML.ParserError || rethrow()
        config_error("TOML syntax error in $file:\n" * sprint(showerror, e))
    end

    top = Section("", raw)
    for k in keys(raw)
        k in TOP_LEVEL_SECTIONS && continue
        hint = _closest(k, TOP_LEVEL_SECTIONS)
        config_error("unknown section [$k]" * (isnothing(hint) ? "" : " (did you mean [$hint]?)"))
    end

    case = parse_case_section(subsection(top, "case"; required=true), file)
    flow = parse_flow(subsection(top, "flow"; required=true), dir)
    dim = length(flow.freestream)
    dim in (2, 3) || config_error("[flow] `freestream` must have 2 or 3 entries (sets the problem dimension)")

    solver = parse_solver(subsection(top, "solver"; required=true), dir)
    grid = parse_grid(subsection(top, "grid"; required=true), dim, solver)
    bodies = parse_bodies(get(raw, "body", nothing), dim, dir, solver)
    ic = parse_initial_condition(subsection(top, "initial_condition"), dim, solver)
    reference = parse_reference(subsection(top, "reference"))
    output = parse_output(subsection(top, "output"), dir, solver)

    CaseConfig(file, dir, dim, case, flow, grid, bodies, solver, ic, reference, output)
end

# --- [case] -----------------------------------------------------------------

function parse_case_section(s, file)
    name = get_string(s, "name", first(splitext(basename(dirname(file)))))
    backend = get_string(s, "backend", "CPU"; allowed=("CPU", "CUDA", "AMDGPU"))
    check_unknown_keys(s)
    (; name, backend)
end

# --- [flow] -----------------------------------------------------------------

function parse_flow(s, dir)
    Re = get_float(s, "Re"; min=0, strict_min=true)
    freestream = get_vector(s, "freestream")
    ramp_time = get_float(s, "ramp_time", 0.0; min=0)
    # Optional user file defining `freestream(t)`; overrides `freestream`'s values
    # (which still sets the dimension).
    file = haskey(s, "file") ? get_path(s, "file", dir) : nothing
    check_unknown_keys(s)
    (; Re, freestream, ramp_time, file)
end

# --- [solver] ---------------------------------------------------------------

function parse_solver(s, dir)
    formulation = get_string(s, "formulation", "FastIBPM"; allowed=("FastIBPM", "IBPM"))
    dt = get_float(s, "dt"; min=0, strict_min=true)
    t_final = get_float(s, "t_final"; min=0, strict_min=true)
    delta = get_string(s, "delta", "Yang3S2"; allowed=("Yang3S", "Yang3S2"))
    ab_order = get_int(s, "ab_order", 2; min=1)
    preconditioner = get_string(s, "preconditioner", "auto"; allowed=("auto", "none", "jacobi"))
    # Absolute tolerance of the iterative coupling solve; absent → solver default.
    coupling_tol = haskey(s, "coupling_tol") ?
                   get_float(s, "coupling_tol"; min=0, strict_min=true) : nothing
    restart_from = haskey(s, "restart_from") ? get_path(s, "restart_from", dir) : nothing
    check_unknown_keys(s)
    !isnothing(restart_from) && formulation != "FastIBPM" &&
        config_error("[solver] `restart_from` is only supported with formulation = \"FastIBPM\"")
    (; formulation, dt, t_final, delta, ab_order, preconditioner, coupling_tol, restart_from)
end

# --- [grid] -----------------------------------------------------------------

function parse_grid(s, dim, solver)
    type = get_string(s, "type", "uniform"; allowed=("uniform", "stretched"))
    g = if type == "uniform"
        h = get_float(s, "h"; min=0, strict_min=true)
        limits = get_limits(s, "limits"; len=dim)
        levels = get_int(s, "levels", 1; min=1)
        (; type, h, limits, levels)
    else
        solver.formulation == "IBPM" ||
            config_error("[grid] type = \"stretched\" requires [solver] formulation = \"IBPM\"")
        h = get_float(s, "h"; min=0, strict_min=true)
        core = get_limits(s, "core"; len=dim)
        extent = get_limits(s, "extent"; len=dim)
        growth = get_float(s, "growth"; min=1)
        max_dx = get_float(s, "max_dx", Inf; min=0, strict_min=true)
        all(extent[:, 1] .<= core[:, 1]) && all(core[:, 2] .<= extent[:, 2]) ||
            config_error("[grid] `core` must lie inside `extent`")
        (; type, h, core, extent, growth, max_dx, levels=1)
    end
    check_unknown_keys(s)
    g
end

# --- [[body]] ---------------------------------------------------------------

const SHAPES_2D = ("cylinder", "plate", "naca4", "points_file")
const SHAPES_3D = ("points_file",)  # sphere etc. come with the 3D work

function parse_bodies(raw, dim, dir, solver)
    isnothing(raw) && config_error("at least one [[body]] section is required")
    raw isa AbstractVector ||
        config_error("bodies must be written as [[body]] (double brackets), even for a single body")
    bodies = NamedTuple[]
    for (k, b) in enumerate(raw)
        push!(bodies, parse_body(Section("body #$k", b), k, dim, dir))
    end
    names = [b.name for b in bodies]
    allunique(names) || config_error("[[body]] names must be unique, got $(names)")
    if any(b -> b.model == "deformable", bodies)
        length(bodies) == 1 ||
            config_error("a deformable body cannot (yet) be combined with other bodies")
        solver.formulation == "FastIBPM" ||
            config_error("deformable bodies require [solver] formulation = \"FastIBPM\"")
        isnothing(solver.restart_from) ||
            config_error("[solver] `restart_from` is not supported for deformable bodies")
    end
    bodies
end

function parse_body(s, k, dim, dir)
    name = get_string(s, "name", "body$k")
    s = Section("body \"$name\"", s.dict, s.used)
    model = get_string(s, "model", "rigid"; allowed=("rigid", "deformable"))
    shapes = dim == 2 ? SHAPES_2D : SHAPES_3D
    shape = get_string(s, "shape"; allowed=shapes)
    spacing = get_float(s, "spacing", 2.0; min=0, strict_min=true)

    geom = parse_shape(s, shape, dim, dir)

    motion = parse_motion(subsection(s, "motion"), dim, dir)
    structure = parse_structure(subsection(s, "structure"), model)

    if model == "deformable"
        shape == "plate" || config_error("[$(s.name)] deformable bodies must have shape = \"plate\"")
        isnothing(motion) || config_error("[$(s.name)] a deformable body cannot have a [body.motion]")
    end
    check_unknown_keys(s)
    (; name, model, shape, spacing, geom, motion, structure)
end

function parse_shape(s, shape, dim, dir)
    if shape == "cylinder"
        radius = get_float(s, "radius"; min=0, strict_min=true)
        center = get_vector(s, "center", zeros(dim); len=dim)
        (; radius, center)
    elseif shape == "plate"
        length_ = get_float(s, "length"; min=0, strict_min=true)
        angle = deg2rad(get_float(s, "angle", 0.0))
        has_c, has_s = haskey(s, "center"), haskey(s, "start")
        has_c && has_s && config_error("[$(s.name)] give either `center` or `start` for a plate, not both")
        # `start` is the leading end; `center` the mid-point. Default: centered at origin.
        dir_ = [cos(angle), sin(angle)]
        start = if has_s
            get_vector(s, "start"; len=dim)
        else
            get_vector(s, "center", zeros(dim); len=dim) .- (length_ / 2) .* dir_
        end
        (; length=length_, angle, start)
    elseif shape == "naca4"
        code = get_string(s, "code")
        (length(code) == 4 && all(isdigit, code)) ||
            config_error("[$(s.name)] `code` must be a 4-digit NACA code such as \"0012\"")
        chord = get_float(s, "chord", 1.0; min=0, strict_min=true)
        angle_of_attack = deg2rad(get_float(s, "angle_of_attack", 0.0))
        pivot_chord = get_float(s, "pivot", 0.5; min=0, max=1)
        center = get_vector(s, "center", zeros(dim); len=dim)
        closed_te = get_bool(s, "closed_trailing_edge", false)
        (; code, chord, angle_of_attack, pivot_chord, center, closed_te)
    elseif shape == "points_file"
        file = get_path(s, "file", dir)
        closed = get_bool(s, "closed", true)
        (; file, closed)
    end
end

const MOTION_TYPES = ("translate", "oscillate", "pitch_heave", "user")

function parse_motion(s, dim, dir)
    isnothing(s) && return nothing
    type = get_string(s, "type"; allowed=MOTION_TYPES)
    m = if type == "translate"
        velocity = get_vector(s, "velocity"; len=dim)
        (; type, velocity)
    elseif type == "oscillate"
        amplitude = get_vector(s, "amplitude"; len=dim)
        frequency = get_float(s, "frequency"; min=0)
        phase = deg2rad(get_float(s, "phase", 0.0))
        (; type, amplitude, frequency, phase)
    elseif type == "pitch_heave"
        dim == 2 || config_error("[$(s.name)] pitch_heave motion is 2D only (for now)")
        heave_amplitude = get_float(s, "heave_amplitude", 0.0)
        pitch_amplitude = deg2rad(get_float(s, "pitch_amplitude", 0.0))
        frequency = get_float(s, "frequency"; min=0)
        phase = deg2rad(get_float(s, "phase", 90.0))
        pivot = haskey(s, "pivot") ? get_vector(s, "pivot"; len=dim) : nothing
        (; type, heave_amplitude, pitch_amplitude, frequency, phase, pivot)
    else # user
        file = get_path(s, "file", dir)
        pivot = haskey(s, "pivot") ? get_vector(s, "pivot"; len=dim) : nothing
        # Free-form parameters forwarded to the user's functions as `params`.
        params = let p = subsection(s, "parameters")
            isnothing(p) ? (;) : (; (Symbol(k) => v for (k, v) in p.dict)...)
        end
        (; type, file, pivot, params)
    end
    check_unknown_keys(s)
    m
end

function parse_structure(s, model)
    if model == "rigid"
        isnothing(s) || config_error("[$(s.name)] is only valid for model = \"deformable\"")
        return nothing
    end
    isnothing(s) && config_error("a deformable body needs a [body.structure] section")
    mass = get_float(s, "mass"; min=0, strict_min=true)
    bending_stiffness = get_float(s, "bending_stiffness"; min=0)
    extensional_stiffness = get_float(s, "extensional_stiffness"; min=0)
    clamped = get_bool(s, "clamped", true)
    initial_angle = deg2rad(get_float(s, "initial_angle", 0.0))
    check_unknown_keys(s)
    (; mass, bending_stiffness, extensional_stiffness, clamped, initial_angle)
end

# --- [initial_condition] ----------------------------------------------------

function parse_initial_condition(s, dim, solver)
    isnothing(s) && return (; type="none")
    type = get_string(s, "type", "none"; allowed=("none", "vortex_blob"))
    ic = if type == "vortex_blob"
        dim == 2 || config_error("[initial_condition] vortex_blob is 2D only (for now)")
        solver.formulation == "FastIBPM" ||
            config_error("[initial_condition] vortex_blob requires formulation = \"FastIBPM\"")
        center = get_vector(s, "center"; len=dim)
        radius = get_float(s, "radius"; min=0, strict_min=true)
        amplitude = get_float(s, "amplitude")
        (; type, center, radius, amplitude)
    else
        (; type)
    end
    check_unknown_keys(s)
    ic
end

# --- [reference] ------------------------------------------------------------

function parse_reference(s)
    isnothing(s) && return (; velocity=1.0, length=1.0)
    velocity = get_float(s, "velocity", 1.0; min=0, strict_min=true)
    length_ = get_float(s, "length", 1.0; min=0, strict_min=true)
    check_unknown_keys(s)
    (; velocity, length=length_)
end

# --- [output] ---------------------------------------------------------------

function parse_output(s, dir, solver)
    s = something(s, Section("output", Dict{String,Any}()))
    directory = let d = get_string(s, "directory", "output")
        isabspath(d) ? d : joinpath(dir, d)
    end
    history_every = get_int(s, "history_every", 1; min=1)
    snapshot_every = get_int(s, "snapshot_every", 100; min=0)
    allowed = solver.formulation == "IBPM" ? ("omega", "u", "p") : ("omega", "u")
    fields = get_string_list(s, "fields", ["omega", "u"]; allowed)
    log_every = get_int(s, "log_every", 100; min=0)
    restart_every = get_int(s, "restart_every", 0; min=0)
    overwrite = get_bool(s, "overwrite", true)
    check_unknown_keys(s)
    restart_every > 0 && solver.formulation != "FastIBPM" &&
        config_error("[output] `restart_every` is only supported with formulation = \"FastIBPM\"")
    (; directory, history_every, snapshot_every, fields, log_every, restart_every, overwrite)
end
