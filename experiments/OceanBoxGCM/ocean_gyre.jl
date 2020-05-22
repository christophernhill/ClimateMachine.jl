#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.HydrostaticBoussinesq

using Test

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function config_simple_box(FT, N, resolution, dimensions; BC = nothing)
    if BC == nothing
        problem = OceanGyre{FT}(dimensions...)
    else
        problem = OceanGyre{FT}(dimensions...; BC = BC)
    end

    _grav::FT = grav(param_set)
    cʰ = sqrt(_grav * problem.H) # m/s
    model = HydrostaticBoussinesqModel{FT}(param_set, problem, cʰ = cʰ)

    config = ClimateMachine.OceanBoxGCMConfiguration(
        "ocean_gyre",
        N,
        resolution,
        model,
    )

    return config
end

function run_ocean_gyre(; imex::Bool = false, BC = nothing)
    FT = Float64

    # DG polynomial order
    N = Int(4)

    # Domain resolution and size
    Nˣ = Int(20)
    Nʸ = Int(20)
    Nᶻ = Int(20)
    resolution = (Nˣ, Nʸ, Nᶻ)

    Lˣ = 4e6    # m
    Lʸ = 4e6    # m
    H = 1000   # m
    dimensions = (Lˣ, Lʸ, H)

    outpdir = "output"
    timestart = FT(0)    # s
    timeout = FT(0.25 * 86400) # s
    timeend = FT(864000) # s
    # timeend = FT(20) # s
    dt = FT(800)    # s

    if imex
        solver_type =
            ClimateMachine.IMEXSolverType(linear_model = LinearHBModel)
    else
        solver_type = ClimateMachine.ExplicitSolverType(
            solver_method = LSRK144NiegemannDiehlBusch,
        )
    end

    driver_config = config_simple_box(FT, N, resolution, dimensions; BC = BC)

    grid = driver_config.grid
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1)
    exp_filter = ExponentialFilter(grid, 1, 8)
    modeldata = (vert_filter = vert_filter, exp_filter = exp_filter)

    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        init_on_cpu = true,
        ode_dt = dt,
        Courant_number = 0.4,
        ode_solver_type = solver_type,
        modeldata = modeldata,
    )

    mkpath(outpdir)
    ClimateMachine.Settings.vtk = "never"
    # vtk_interval = ceil(Int64, timeout / solver_config.dt)
    # ClimateMachine.Settings.vtk = "$(vtk_interval)steps"

    ClimateMachine.Settings.diagnostics = "never"
    # diagnostics_interval = ceil(Int64, timeout / solver_config.dt)
    # ClimateMachine.Settings.diagnostics = "$(diagnostics_interval)steps"

    ## Create a callback to report state statistics for main MPIStateArrays
    ## every ntFreq timesteps.
    ntFreq=10
    cb=ClimateMachine.StateCheck.StateCheck.sccreate( [ (solver_config.Q,"Q",),
                                              (solver_config.dg.state_auxiliary,"s_aux",),
                                              (solver_config.dg.state_gradient_flux,"s_gflux",) ],
                                            ntFreq; prec=12 )

    result = ClimateMachine.invoke!(solver_config; user_callbacks=[cb] )

    Qnd=reshape(solver_config.Q.realdata,(5,5,5,4,20,20,20));
    Gnd=reshape(solver_config.dg.grid.vgeo,(5,5,5,16,20,20,20));
    tval=Qnd[1,1,:,4,:,1,1][:];
    xval=Gnd[:,1,1,13,1,1,:][:]
    yval=Gnd[1,:,1,14,1,:,1][:]
    zval=Gnd[1,1,:,15,:,1,1][:]
    tval_cpu=ones( size(tval) )
    copyto!(tval_cpu, tval)
    typeof(tval_cpu)
    zval_cpu=ones(size(zval))
    copyto!(zval_cpu, zval)
    println("xval =",xval)
    println("yval =",yval)
    println("zval =",zval)


    @test true
end

@testset "$(@__FILE__)" begin
    boundary_conditions = [
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
    ]

        boundary_conditions = [
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceStressNoForcing(),
#           ClimateMachine.HydrostaticBoussinesq.OceanSurfaceNoStressNoForcing(),
        ),
     ]

    for BC in boundary_conditions
        run_ocean_gyre(imex = false, BC = BC)
    end
end
