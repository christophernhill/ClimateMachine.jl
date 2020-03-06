export StormerVerlet, StromerVerletHEVI

abstract type AbstractStormerVerlet <: ODEs.AbstractODESolver end

ODEs.updatedt!(sv::AbstractStormerVerlet, dt) = sv.dt[1] = dt
#=
"""
    ODESolvers.dostep!(Q, sv::StormerVerlet, p, timeend::Real,
                       adjustfinalstep::Bool)
Use the 2N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time, to the time `timeend`. If `adjustfinalstep == true` then
`dt` is adjusted so that the step does not take the solution beyond the
`timeend`.
"""
function ODEs.dostep!(Q, sv::AbstractStormerVerlet, p, timeend::Real,
                      adjustfinalstep::Bool, slow_δ, slow_rv_dQ, slow_rka)
  time, dt = sv.t[1], sv.dt[1]
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
  end
  @assert dt > 0

  ODEs.dostep!(Q, sv, p, time, dt, slow_δ, slow_rv_dQ, slow_rka)

  if dt == sv.dt[1]
    sv.t[1] += dt
  else
    sv.t[1] = timeend
  end

end
=#

"""
    LowStorageRungeKutta2N(f, RKA, RKB, RKC, Q; dt, t0 = 0)
This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,
```math
  \\dot{Q} = f(Q, t)
```
with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.
The constructor builds a low-storage Runge-Kutta scheme using 2N
storage based on the provided `RKA`, `RKB` and `RKC` coefficient arrays.
The available concrete implementations are:
  - [`LSRK54CarpenterKennedy`](@ref)
  - [`LSRK144NiegemannDiehlBusch`](@ref)
"""
struct StormerVerlet{N, T, RT, AT} <: AbstractStormerVerlet
  "time step"
  dt::RT
  "time"
  t::RT
  "rhs function"
  rhs!

  mask_a
  mask_b

  dQ::AT
  function StormerVerlet(rhs!::TimeScaledRHS{N,RT} where {RT}, mask_a, mask_b, Q::AT; dt=0, t0=0) where {N,AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)

    dQ = similar(Q)
    fill!(dQ, 0)

    new{N, T, RT, AT}(dt, t0, rhs!, mask_a, mask_b, dQ)
  end
end

"""
    ODESolvers.dostep!(Q, lsrk::LowStorageRungeKutta2N, p, time::Real,
                       dt::Real, [slow_δ, slow_rv_dQ, slow_scaling])
Use the 2N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time `time` to final time `time + dt`.
If the optional parameter `slow_δ !== nothing` then `slow_rv_dQ * slow_δ` is
added as an additional ODE right-hand side source. If the optional parameter
`slow_scaling !== nothing` then after the final stage update the scaling
`slow_rv_dQ *= slow_scaling` is performed.
"""
function ODEs.dostep!(Q, sv::StormerVerlet{1,T,RT,AT} where {T,RT,AT}, p, time::Real,
                      dt::Real, nsteps::Int, slow_δ, slow_rv_dQ, slow_rka)

  rhs!, dQ = sv.rhs!, sv.dQ

  Qa = @view(Q.realdata[:,sv.mask_a,:])
  Qb = @view(Q.realdata[:,sv.mask_b,:])
  dQa = @view(dQ.realdata[:,sv.mask_a,:])
  dQb = @view(dQ.realdata[:,sv.mask_b,:])
  slow_rv_dQa = @view(slow_rv_dQ[:,sv.mask_a,:])
  slow_rv_dQb = @view(slow_rv_dQ[:,sv.mask_b,:])



  # do a half step
  rhs!(dQ, Q, p, time, increment = false)

  if slow_δ === nothing
    Qa .+= dQa .* dt/2
  else
    Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* dt/2
  end
  time += dt/2

  for i = 1:nsteps
    rhs!(dQ, Q, p, time, increment = false)
    if slow_δ === nothing
      Qb .+= dQb .* dt
    else
      Qb .+= (dQb .+ slow_rv_dQb .* slow_δ) .* dt
    end
    time += dt

    rhs!(dQ, Q, p, time, increment = false)
    if i < nsteps
      if slow_δ === nothing
        Qa .+= dQa .* dt
      else
        Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* dt
      end
      time += dt
    else
      if slow_δ === nothing
        Qa .+= dQa .* dt/2
      else
        Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* dt/2
      end
      time += dt/2
    end
  end
  if slow_rka !== nothing
    slow_rv_dQ .*= slow_rka
  end
end

"""
    ODESolvers.dostep!(Q, lsrk::LowStorageRungeKutta2N, p, time::Real,
                       dt::Real, [slow_δ, slow_rv_dQ, slow_scaling])
Use the 2N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time `time` to final time `time + dt`.
If the optional parameter `slow_δ !== nothing` then `slow_rv_dQ * slow_δ` is
added as an additional ODE right-hand side source. If the optional parameter
`slow_scaling !== nothing` then after the final stage update the scaling
`slow_rv_dQ *= slow_scaling` is performed.
"""
function ODEs.dostep!(Q, sv::StormerVerlet{2,T,RT,AT} where {T,RT,AT}, p, time::Real,
                      dt::Real, nsteps::Int, slow_δ, slow_rv_dQ, slow_rka)

  rhs!, dQ = sv.rhs!, sv.dQ

  Qa = @view(Q.realdata[:,sv.mask_a,:])
  Qb = @view(Q.realdata[:,sv.mask_b,:])
  dQa = @view(dQ.realdata[:,sv.mask_a,:])
  dQb = @view(dQ.realdata[:,sv.mask_b,:])
  slow_rv_dQa = @view(slow_rv_dQ[:,sv.mask_a,:])
  slow_rv_dQb = @view(slow_rv_dQ[:,sv.mask_b,:])


  # do a half step
  rhs!(dQ, Q, p, time, 2, increment = false) #Thermo
  if slow_δ === nothing
    @. Qa += dQa * dt/2
  else
    @. Qa += (dQa + slow_rv_dQa * slow_δ) * dt/2
  end
  time += dt/2

  for i = 1:nsteps
    rhs!(dQ, Q, p, time, 1, increment = false) #Momentum
    if slow_δ === nothing
      @. Qb += dQb * dt
    else
      @. Qb += (dQb + slow_rv_dQb * slow_δ) * dt
    end
    time += dt

    rhs!(dQ, Q, p, time, 2, increment = false) #Thermo
    if i < nsteps
      if slow_δ === nothing
        @. Qa += dQa * dt
      else
        @. Qa += (dQa + slow_rv_dQa * slow_δ) * dt
      end
      time += dt
    else
      if slow_δ === nothing
        @. Qa += dQa * dt/2
      else
        @. Qa += (dQa + slow_rv_dQa * slow_δ) * dt/2
      end
      time += dt/2
    end
  end
  if slow_rka !== nothing
    slow_rv_dQ .*= slow_rka
  end
end


struct StormerVerletHEVI{T, RT, AT} <: AbstractStormerVerlet
  "time step"
  dt::RT
  "time"
  t::RT
  "rhs function"
  rhs_h!
  rhs_v!

  A_v

  mask_a
  mask_b

  dQ::AT
  function StormerVerletHEVI(rhs_h!, rhs_v!, mask_a, mask_b, Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)

    A_v = banded_matrix(rhs_v!, similar(Q), similar(Q))

    dQ = similar(Q)
    fill!(dQ, 0)

    new{T, RT, AT}(dt, t0, rhs_h!, rhs_v!, A_v, mask_a, mask_b, dQ)
  end
end

"""
    ODESolvers.dostep!(Q, lsrk::LowStorageRungeKutta2N, p, time::Real,
                       dt::Real, [slow_δ, slow_rv_dQ, slow_scaling])
Use the 2N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time `time` to final time `time + dt`.
If the optional parameter `slow_δ !== nothing` then `slow_rv_dQ * slow_δ` is
added as an additional ODE right-hand side source. If the optional parameter
`slow_scaling !== nothing` then after the final stage update the scaling
`slow_rv_dQ *= slow_scaling` is performed.
"""
function ODEs.dostep!(Q, sv::StormerVerletHEVI, p, time::Real,
                      dt::Real, nsteps::Int, slow_δ, slow_rv_dQ, slow_rka)

  rhs_h!, rhs_v!, dQ, A_v = sv.rhs_h!, sv.rhs_v!, sv.dQ, sv.A_v

  Qa = @view(Q.realdata[:,sv.mask_a,:])
  Qb = @view(Q.realdata[:,sv.mask_b,:])
  dQa = @view(dQ.realdata[:,sv.mask_a,:])
  dQb = @view(dQ.realdata[:,sv.mask_b,:])
  slow_rv_dQa = @view(slow_rv_dQ[:,sv.mask_a,:])
  slow_rv_dQb = @view(slow_rv_dQ[:,sv.mask_b,:])


  # do a half step
  banded_matrix_vector_product!(rhs_v!, A_v, dQ,Q)
  rhs_h!(dQ, Q, p, time, increment = true)
  if slow_δ === nothing
    Qa .+= dQa .* dt/2
  else
    Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* dt/2
  end
  time += dt/2

  for i = 1:nsteps
    banded_matrix_vector_product!(rhs_v!, A_v, dQ,Q)
    rhs_h!(dQ, Q, p, time, increment = true)
    if slow_δ === nothing
      Qb .+= dQb .* dt
    else
      Qb .+= (dQb .+ slow_rv_dQb .* slow_δ) .* dt
    end
    time += dt

    banded_matrix_vector_product!(rhs_v!, A_v, dQ,Q)
    rhs_h!(dQ, Q, p, time, increment = true)
    if i < nsteps
      if slow_δ === nothing
        Qa .+= dQa .* dt
      else
        Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* dt
      end
      time += dt
    else
      if slow_δ === nothing
        Qa .+= dQa .* dt/2
      else
        Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* dt/2
      end
      time += dt/2
    end
  end
  if slow_rka !== nothing
    slow_rv_dQ .*= slow_rka
  end
end