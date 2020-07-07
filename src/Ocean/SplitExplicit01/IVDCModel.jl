# Linear model equations, for split-explicit ocean model implicit vertical diffusion 
# convective adjustment step.
#
# In this version the operator is tweked to be the indentity for testing

using ClimateMachine.DGMethods.NumericalFluxes:
    NumericalFluxFirstOrder, NumericalFluxSecondOrder, NumericalFluxGradient

"""
 IVDCModel{M} <: BalanceLaw

 This code defines DG `BalanceLaw` terms for an operator, L, that is evaluated from iterative 
 implicit solver to solve an equation of the form 

 (L + 1/Δt) ϕ^{n+1} = ϕ^{n}/Δt

  where L is a vertical diffusion operator with a spatially varying diffusion
  coefficient.

 # Usage
 
 parent_model  = OceanModel{FT}(prob...)
 linear_model  = IVDCModel( parent_model )
 	
"""

# Create a new child linear model instance, attached to whatever parent
# BalanceLaw instantiates this.
# (Not sure we need parent, but maybe we will get some parameters from it)
struct IVDCModel{M} <: AbstractOceanModel
  parent_om::M
  function IVDCModel(parent_om::M) where {M}
    return new{M}(parent_om)
  end
end

## struct BarotropicModel{M} <: AbstractOceanModel
##     baroclinic::M
##     function BarotropicModel(baroclinic::M) where {M}
##         return new{M}(baroclinic)
##     end
## end



"""
 Set model state variables and operators
"""

# State variable and initial value, just one for now, θ
 
vars_state_conservative(m::IVDCModel,FT) = @vars(θ::FT)

function init_state_conservative!(
    m::IVDCModel,
    Q::Vars,
    A::Vars,
    coords,
    t,
)
  @inbounds begin
    Q.θ = -0
  end
  return nothing
end

vars_state_auxiliary(m::IVDCModel, FT) = @vars(θ_rhs::FT)
init_state_auxiliary!(m::IVDCModel, _...) = nothing


# Variables and operations used in differentiating first derivatives

vars_state_gradient(m::IVDCModel, FT) = @vars(∇θ::FT)

@inline function compute_gradient_argument!(
    m::IVDCModel,
    G::Vars,
    Q::Vars,
    A,
    t,
)
    G.∇θ = Q.θ
#    G.∇θ = -0

    return nothing
end


# Variables and operations used in differentiating second derivatives
 
vars_state_gradient_flux(m::IVDCModel, FT) = @vars(κ∇θ::SVector{3, FT})

@inline function compute_gradient_flux!(
    m::IVDCModel,
    D::Vars,
    G::Grad,
    Q::Vars,
    A::Vars,
    t,
)

    κ = diffusivity_tensor(m, G.∇θ[3])
    D.κ∇θ = -κ * G.∇θ
#    D.κ∇θ = -κ * G.∇θ * 0.

    return nothing
end


@inline function diffusivity_tensor(m::IVDCModel, ∂θ∂z)
  # ∂θ∂z < 0 ? κ = (@SVector [m.κʰ, m.κʰ, 1000 * m.κᶻ]) : κ =
    κᶻ=m.parent_om.κᶻ
    κᶻ=1000
    κᶻ=0.1
    ∂θ∂z < 0 ? κ = (@SVector [0, 0, 1 * κᶻ]) : κ =
        (@SVector [0, 0, κᶻ])

    return Diagonal(κ)
end

# Function to apply I to state variable

@inline function source!(
    m::IVDCModel,
    S::Vars,
    Q::Vars,
    D::Vars,
    A::Vars,
    t,
    direction,
)
    @inbounds begin
     # S!.θ=Q.θ/m.parent_om.dt_slow
     S.θ=Q.θ
     # S.θ=0
    end

    return nothing
end

## Numerical fluxes and boundaries

function flux_first_order!(::IVDCModel, _...) end

function flux_second_order!(
    ::IVDCModel,
    F::Grad,
    S::Vars,
    D::Vars,
    H::Vars,
    A::Vars,
    t,
)
    F.θ += D.κ∇θ

end

function wavespeed(m::IVDCModel, n⁻, _...)
    C = abs(SVector(m.parent_om.cʰ, m.parent_om.cʰ, m.parent_om.cᶻ)' * n⁻)
    return C
end


function boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient, CentralNumericalFluxGradient},
    m::IVDCModel,
    Q⁺,
    A⁺,
    n,
    Q⁻,
    A⁻,
    bctype,
    t,
    _...,
)
    Q⁺.θ = Q⁻.θ

    return nothing
end

###    From -  function numerical_boundary_flux_gradient! , DGMethods/NumericalFluxes.jl
###    boundary_state!(
###        numerical_flux,
###        balance_law,
###        state_conservative⁺,
###        state_auxiliary⁺,
###        normal_vector,
###        state_conservative⁻,
###        state_auxiliary⁻,
###        bctype,
###        t,
###        state1⁻,
###        aux1⁻,
###    )


function boundary_state!(
    nf::Union{NumericalFluxSecondOrder,CentralNumericalFluxSecondOrder},
    m::IVDCModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    bctype,
    t,
    _...,
)
    Q⁺.θ = Q⁻.θ
    D⁺.κ∇θ = n⁻ * -0
    # D⁺.κ∇θ = -D⁻.κ∇θ

    return nothing
end

###    boundary_state!(
###        numerical_flux,
###        balance_law,
###        state_conservative⁺,
###        state_gradient_flux⁺,
###        state_auxiliary⁺,
###        normal_vector,
###        state_conservative⁻,
###        state_gradient_flux⁻,
###        state_auxiliary⁻,
###        bctype,
###        t,
###        state1⁻,
###        diff1⁻,
###        aux1⁻,
###    )

