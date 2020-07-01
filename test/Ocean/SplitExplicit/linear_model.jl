# Linear model equations, for split-explicit ocean model implicit vertical diffusion 
# convective adjustment step.
"""
 om_ivdc_lm{M} <: BalanceLaw

 This code defines DG `BalanceLaw` terms for an operator, L, that is evaluated from iterative 
 implicit solver to solve an equation of the form 

 (L + 1/Δt) ϕ^{n+1} = ϕ^{n}/Δt

 The L operator here is written to work with the state from a parent baroclinic ocean model (or equivalent)
 that defines a prognostic variable θ and a diffusivity κ, on a DG balance law model mesh and with both 
 θ and κ fully varying in space and time.

 ** NEED TO FIND OUT ABOUT Δt PART AND HOW IT IS INCLUDED

 # Usage
 
 parent_model  = OceanModel{FT}(prob...)
 linear_model  = om_ivdc_lin_mod( parent_model )
 	
"""

# Create a new child linear model instance, attached to whatever parent
# BalanceLaw instantiates this.
# (No longer sure we need parent, but maybe we will get some parameters from it)
struct om_ivdc_lm{M} <: BalanceLaw
  parent_om::M
  function om_ivdc_lm{parent_om::M} where {M}
    return new{M}(parent_om)
  end
end


"""
 Map child model state to parent variables
"""

# State variable, just one for now, θ
vars_state_conservative(lm:om_ivdc_lm, FT) = @vars(θ::FT)

# "Gradiented" variables, θ
vars_state_gradient(lm:om_ivdc_lm, FT) = @vars(θ::FT)

# Intermediate first order flux variables, κ∇θ
vars_state_gradient_flux(lm::om_ivdc_lm, FT) = @vars(κ∇θ::SVector{3, T})

# All other vars_() can be empty for this linear model
vars_state_auxiliary(lm::om_ivdc_lm, FT) = @vars()
vars_integrals(lm::om_ivdc_lm, FT) = @vars()

"""
    compute_gradient_argument!(::om_ivdc_lm)

copy θ to var_gradient
this computation is done pointwise at each nodal point

# arguments:
- `m`: model in this case HBModel
- `G`: array of gradient variables (to be computed in a kernel)
- `Q`: array of state variables
- `A`: array of aux variables
- `t`: time, not used
"""
@inline function compute_gradient_argument!(
    m::om_ivdc_lm,
    G!::Vars,
    Q::Vars,
    A,
    t,
)
    G!.θ = Q.θ

    return nothing
end


"""
    compute_gradient_flux!(::om_ivdc_lm)

copy gradient of θ to variable to be differentiated
"""
@inline function compute_gradient_flux!(
    m::om_ivdc_lm,
    D!::Vars,
    G::Grad,
    Q::Vars,
    A::Vars,
    t,
)

    κ = diffusivity_tensor(m, G.θ[3])
    D!.κ∇θ = κ * G.θ

    return nothing
end

@inline function source!(
    m::om_ivdc_lm
    S!::Vars,
    Q::Vars,
    D::Vars,
    A::Vars,
    t::Real,
    direction,
)
    @inbounds begin
     S!.θ=Q.θ/m.parent_om.dt_slow
    end

    return nothing
end


@inline function diffusivity_tensor(m::om_ivdc_lm, ∂θ∂z)
  # ∂θ∂z < 0 ? κ = (@SVector [m.κʰ, m.κʰ, 1000 * m.κᶻ]) : κ =
    ∂θ∂z < 0 ? κ = (@SVector [0, 0, 1 * m.κᶻ]) : κ =
        (@SVector [0, 0, m.κᶻ])

    return Diagonal(κ)
end

# Need to add something to control hidden numerical flux computations
# and penalty function

# What about bc's - these are zero flux, so probably OK.

# What about 3d mesh but 1d problem?


