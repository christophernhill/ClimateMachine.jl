abstract type RadiationModel
end

struct NoRadiation <: RadiationModel
end
vars_aux(m::NoRadiation) = ()
function flux!(m::NoRadiation, flux::Grad, state::State, diffusive::State, auxstate::State, t::Real)
end
function preodefun!(m::NoRadiation, auxstate::State, state::State, t::Real)
end

struct StevensRadiation <: RadiationModel
end
vars_aux(m::StevensRadiation) = (:z, :zero_to_z, :z_to_inf)
function flux!(m::StevensRadiation, flux::Grad, state::State, diffusive::State, auxstate::State, t::Real)
    T = eltype(flux)

    z_i = T(840)  # Start with constant inversion height of 840 meters then build in check based on q_tot
    Δz_i = max(aux.z - z_i, zero(T))

    # Constants
    α_z = T(1)
    ρ_i = T(1.22)
    D_subsidence = T(3.75e-6)
    term1 = T(70) * exp(-aux.z_to_inf) 
    term2 = T(22) * exp(-aux.zero_to_z)
    term3 = ρ_i * T(cp_d) * D_subsidence * α_z * ((cbrt(Δz_i))^4 / 4 + z_i * cbrt(Δz_i))
    F_rad = term1 + term2 + term3 

    flux.ρe -= SVector(T(0), T(0), state.ρ * F_rad)
end
function preodefun!(m::StevensRadiation, auxstate::State, state::State, t::Real)
end
