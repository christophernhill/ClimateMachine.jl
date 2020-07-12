using ClimateMachine
ClimateMachine.cli()

using ClimateMachine.BalanceLaws: vars_state_conservative, vars_state_auxiliary
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates: flattenednames
using ClimateMachine.SplitExplicit01
using ClimateMachine.GenericCallbacks
using ClimateMachine.VTK

using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()


import ClimateMachine.SplitExplicit01:
    ocean_init_aux!,
    ocean_init_state!,
    ocean_boundary_state!,
    CoastlineFreeSlip,
    CoastlineNoSlip,
    OceanFloorFreeSlip,
    OceanFloorNoSlip,
    OceanSurfaceNoStressNoForcing,
    OceanSurfaceStressNoForcing,
    OceanSurfaceNoStressForcing,
    OceanSurfaceStressForcing
import ClimateMachine.DGMethods:
    update_auxiliary_state!, update_auxiliary_state_gradient!, vars_state_conservative, vars_state_auxiliary, VerticalDirection

import ClimateMachine.SystemSolvers:
    BatchedGeneralizedMinimalResidual, linearsolve!

# using GPUifyLoops

const ArrayType = ClimateMachine.array_type()

struct SimpleBox{T} <: AbstractOceanProblem
    Lˣ::T
    Lʸ::T
    H::T
    τₒ::T
    λʳ::T
    θᴱ::T
end



@inline function ocean_boundary_state!(
    m::OceanModel,
    p::SimpleBox,
    bctype,
    x...,
)
    if bctype == 1
        ocean_boundary_state!(m, CoastlineNoSlip(), x...)
    elseif bctype == 2
        ocean_boundary_state!(m, OceanFloorNoSlip(), x...)
    elseif bctype == 3
        ocean_boundary_state!(m, OceanSurfaceStressForcing(), x...)
    end
end

@inline function ocean_boundary_state!(
    m::Continuity3dModel,
    p::SimpleBox,
    bctype,
    x...,
)
   #if bctype == 1
        ocean_boundary_state!(m, CoastlineNoSlip(), x...)
   #end
end

@inline function ocean_boundary_state!(
    m::BarotropicModel,
    p::SimpleBox,
    bctype,
    x...,
)
    return ocean_boundary_state!(m, CoastlineNoSlip(), x...)
end

function ocean_init_state!(p::SimpleBox, Q, A, coords, t)
    @inbounds y = coords[2]
    @inbounds z = coords[3]
    @inbounds H = p.H

    Q.u = @SVector [-0, -0]
    Q.η = -0
    Q.θ = (5 + 4 * cos(y * π / p.Lʸ)) * (1 + z / H)

    return nothing
end

function ocean_init_aux!(m::OceanModel, p::SimpleBox, A, geom)
    FT = eltype(A)
    @inbounds A.y = geom.coord[2]

    # not sure if this is needed but getting weird intialization stuff
    A.w = -0
    A.pkin = -0
    A.wz0 = -0
    A.u_d = @SVector [-0, -0]
    A.ΔGu = @SVector [-0, -0]

    return nothing
end

# A is Filled afer the state
function ocean_init_aux!(
    m::BarotropicModel,
    P::SimpleBox,
    A,
    geom,
)
    @inbounds A.y = geom.coord[2]

    A.Gᵁ = @SVector [-0, -0]
    A.U_c = @SVector [-0, -0]
    A.η_c = -0
    A.U_s = @SVector [-0, -0]
    A.η_s = -0
    A.Δu = @SVector [-0, -0]
    A.η_diag = -0
    A.Δη = -0

    return nothing
end



#################
# RUN THE TESTS #
#################
FT = Float64
vtkpath = "vtk_split"

const timeend = 15 * 24 * 3600 # s
const tout = 24 * 3600 # s
#const timeend = 4 * 300 # s
#const tout = 300 # s

const N = 4
const Nˣ = 20
const Nʸ = 20
const Nᶻ = 20
const Lˣ = 4e6  # m
const Lʸ = 4e6  # m
const H = 1000  # m
Np1=N+1
ne=Nᶻ

xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)

#const cʰ = sqrt(gravity * H)
const cʰ = 1  # typical of ocean internal-wave speed
const cᶻ = 0

#- inverse ratio of additional fast time steps (for weighted average)
#  --> do 1/add more time-steps and average from: 1 - 1/add up to: 1 + 1/add
# e.g., = 1 --> 100% more ; = 2 --> 50% more ; = 3 --> 33% more ...
add_fast_substeps = 2

const τₒ = 1e-1  # (m/s)^2
#const τₒ = 0
const λʳ = 10 // 86400 # m / s
const θᴱ = 10    # K

    ClimateMachine.init()
    mpicomm = MPI.COMM_WORLD

    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
        ll == "WARN" ? Logging.Warn :
        ll == "ERROR" ? Logging.Error : Logging.Info
    logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

    brickrange_2D = (xrange, yrange)
    topl_2D =
        BrickTopology(mpicomm, brickrange_2D, periodicity = (false, false))
    grid_2D = DiscontinuousSpectralElementGrid(
        topl_2D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    brickrange_3D = (xrange, yrange, zrange)
    topl_3D = StackedBrickTopology(
        mpicomm,
        brickrange_3D;
        periodicity = (false, false, false),
        boundary = ((1, 1), (1, 1), (2, 3)),
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topl_3D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    prob = SimpleBox{FT}(Lˣ, Lʸ, H, τₒ, λʳ, θᴱ)
    ## gravity::FT = grav(param_set)
    gravity = grav(param_set)

    model = OceanModel{FT}(prob, grav = gravity, cʰ = cʰ, add_fast_substeps = add_fast_substeps )
    # model = OceanModel{FT}(prob, cʰ = cʰ, fₒ = FT(0), β = FT(0) )
    # model = OceanModel{FT}(prob, cʰ = cʰ, νʰ = FT(1e3), νᶻ = FT(1e-3) )
    # model = OceanModel{FT}(prob, cʰ = cʰ, νʰ = FT(0), fₒ = FT(0), β = FT(0) )

    barotropicmodel = BarotropicModel(model)

    minΔx = min_node_distance(grid_3D, HorizontalDirection())
    minΔz = min_node_distance(grid_3D, VerticalDirection())
    #- 2 horiz directions
    gravity_max_dT = 1 / ( 2 * sqrt(gravity * H) / minΔx )
    dt_fast = minimum([gravity_max_dT])

    #- 2 horiz directions + harmonic visc or diffusion: 2^2 factor in CFL:
    viscous_max_dT = 1 / ( 2 * model.νʰ / minΔx^2 + model.νᶻ / minΔz^2 )/ 4
    diffusive_max_dT = 1 / ( 2 * model.κʰ / minΔx^2 + model.κᶻ / minΔz^2 )/ 4
    dt_slow = minimum([diffusive_max_dT, viscous_max_dT])

    dt_fast = 240
    dt_slow = 5400
  # dt_fast = 300
  # dt_slow = 300
    nout = ceil(Int64, tout / dt_slow)
    dt_slow = tout / nout

    dg = OceanDGModel(
        model,
        grid_3D,
    #   CentralNumericalFluxFirstOrder(),
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    barotropic_dg = DGModel(
        barotropicmodel,
        grid_2D,
    #   CentralNumericalFluxFirstOrder(),
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q_3D = init_ode_state(dg, FT(0); init_on_cpu = true)
    # update_auxiliary_state!(dg, model, Q_3D, FT(0))
    # update_auxiliary_state_gradient!(dg, model, Q_3D, FT(0))

    Q_2D = init_ode_state(barotropic_dg, FT(0); init_on_cpu = true)

    lsrk_ocean = LSRK54CarpenterKennedy(dg, Q_3D, dt = dt_slow, t0 = 0)
    lsrk_barotropic =
        LSRK54CarpenterKennedy(barotropic_dg, Q_2D, dt = dt_fast, t0 = 0)

    odesolver = SplitExplicitLSRK2nSolver(
        lsrk_ocean,
        lsrk_barotropic,
    )

    ### odesolver.slow_solver.rhs!.modeldata.ivdc_dg()
    ###  size(odesolver.slow_solver.rhs!.modeldata.ivdc_Q.θ)
    Grd=odesolver.slow_solver.rhs!.grid.vgeo;
    Qin=odesolver.slow_solver.rhs!.modeldata.ivdc_Q;
    QinXY=reshape( Qin,(Np1*Np1,Np1,ne,ne*ne) )

    dQ=deepcopy(Qin);
    zc=reshape(Grd[:,15,:],(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]

    for iel=1:ne
     for inode=1:Np1
      H=4000.;L=H/10.;A=20.;C=50.;D=2.5;B=8.;E=5.e-4;
      z=zc[inode,iel]
      ft(xx,L)=exp(-xx/L);
      th1(zz)=A*ft.(-zz .+ L, L);
      th2(zz)=C*ft(D*(-zz .+ L), L);
      phi1=th1.(z)
      phi2=th2.(z)

      QinXY[:,inode,iel,:].=phi1 .- phi2 .+ B .+ E .*z;
      # QinXY[:,inode,iel,:].=z
     end
    end

    ivdc_dg=odesolver.slow_solver.rhs!.modeldata.ivdc_dg
    ivdc_dg(dQ,Qin,nothing,0)
    QP=dQ;
    tz=reshape(QP.θ,(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]
    using Plots
    savefig( scatter(tz,zc,label="") , "fooIVDCM-dQ.png" )
    QP=Qin;
    tz=reshape(QP.θ,(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]
    savefig( scatter(tz,zc,label="") , "fooIVDCM-Qin.png" )

    # create linear solver
    bgmSolver=BatchedGeneralizedMinimalResidual(
        ivdc_dg,
        Qin)
    lm!(y,x)=ivdc_dg(y,x,nothing,0;increment=false)
    ivdc_solver_dt=ivdc_dg.balance_law.ivdc_dt

    # run solver( Aop, solv, x_out, b_in )
    dQivdc=deepcopy(Qin)
    Qout=deepcopy(Qin)
    dQivdc.θ.=Qin.θ./ivdc_solver_dt

    Qout.θ.=Qin.θ
    odesolver.slow_solver.rhs!.modeldata.ivdc_dg.state_auxiliary.θ_init.=Qin.θ
    QP=Qout
    tz=reshape(QP.θ,(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]
    savefig( scatter(tz,zc,label="") , "fooBGM-guess.png" )

    QP=dQivdc
    tz=reshape(QP.θ,(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]
    savefig( scatter(tz,zc,label="") , "fooBGM-b.png" )

    # Solve once
    solve_time=@elapsed iters = linearsolve!(lm!, bgmSolver, Qout, dQivdc);
    println("solver iters, time: ",iters, ", ", solve_time)

    # Step forward iteratively a few times
    dQivdc.θ.=Qout.θ./ivdc_solver_dt
    odesolver.slow_solver.rhs!.modeldata.ivdc_dg.state_auxiliary.θ_init.=Qout.θ
    #### Qout.θ.=Qin.θ
    solve_time=@elapsed iters = linearsolve!(lm!, bgmSolver, Qout, dQivdc);
    println("solver iters, time: ",iters, ", ", solve_time)

    # Step forward iteratively a few times
    dQivdc.θ.=Qout.θ./ivdc_solver_dt
    odesolver.slow_solver.rhs!.modeldata.ivdc_dg.state_auxiliary.θ_init.=Qout.θ
    #### Qout.θ.=Qin.θ
    solve_time=@elapsed iters = linearsolve!(lm!, bgmSolver, Qout, dQivdc);
    println("solver iters, time: ",iters, ", ", solve_time)

    # Step forward iteratively a few times
    dQivdc.θ.=Qout.θ./ivdc_solver_dt
    odesolver.slow_solver.rhs!.modeldata.ivdc_dg.state_auxiliary.θ_init.=Qout.θ
    #### Qout.θ.=Qin.θ
    solve_time=@elapsed iters = linearsolve!(lm!, bgmSolver, Qout, dQivdc);
    println("solver iters, time: ",iters, ", ", solve_time)

    # Step forward iteratively a few times
    dQivdc.θ.=Qout.θ./ivdc_solver_dt
    odesolver.slow_solver.rhs!.modeldata.ivdc_dg.state_auxiliary.θ_init.=Qout.θ
    #### Qout.θ.=Qin.θ
    solve_time=@elapsed iters = linearsolve!(lm!, bgmSolver, Qout, dQivdc);
    println("solver iters, time: ",iters, ", ", solve_time)

    QP=Qout
    tz=reshape(QP.θ,(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]
    savefig( scatter(tz,zc,label="") , "fooBGM-x.png" )
