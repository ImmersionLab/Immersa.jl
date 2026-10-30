# Tangential surface actuation on a fixed plate.
#
# `surface_velocity!` keeps the markers at their reference positions and sets
# their velocity u at time t. `params` holds [body.motion.parameters].

function surface_velocity!(u, x_ref, t, params)
    (; amplitude, frequency, phase, s_start, s_end) = params
    chord = x_ref[end] - x_ref[1]
    t̂ = chord / norm(chord)                      # unit tangent
    A = amplitude * sin(2π * frequency * t + phase)
    for j in eachindex(u)
        ŝ = dot(x_ref[j] - x_ref[1], t̂) / norm(chord)   # 0 at leading edge, 1 at trailing edge
        u[j] = s_start <= ŝ <= s_end ? A * t̂ : zero(t̂)
    end
end
