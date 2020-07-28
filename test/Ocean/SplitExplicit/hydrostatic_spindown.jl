#!/usr/bin/env julia --project
using ClimateMachine
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.Ocean
using ClimateMachine.Ocean.HydrostaticBoussinesq
using ClimateMachine.Ocean.ShallowWater
using ClimateMachine.Ocean.SplitExplicit: VerticalIntegralModel
using ClimateMachine.Ocean.OceanProblems

using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.BalanceLaws: vars_state, Prognostic, Auxiliary
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.VTK

using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function run_hydrostatic_spindown(
    vtkpath,
    resolution,
    dimensions,
    timespan;
    coupling = Coupled(),
    dt_slow = 300,
    refDat = (),
)
    mpicomm = MPI.COMM_WORLD
    ArrayType = ClimateMachine.array_type()

    N, Nˣ, Nʸ, Nᶻ = resolution
    Lˣ, Lʸ, H = dimensions
    tout, timeend = timespan

    xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
    yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
    zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)

    brickrange_2D = (xrange, yrange)
    topl_2D = BrickTopology(
        mpicomm,
        brickrange_2D,
        periodicity = (true, true),
        boundary = ((0, 0), (0, 0)),
    )
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
        periodicity = (true, true, false),
        boundary = ((0, 0), (0, 0), (1, 2)),
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topl_3D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    problem = SimpleBox{FT}(Lˣ, Lʸ, H)

    model_3D = HydrostaticBoussinesqModel{FT}(
        param_set,
        problem;
        coupling = coupling,
        cʰ = FT(1),
        αᵀ = FT(0),
        κʰ = FT(0),
        κᶻ = FT(0),
        fₒ = FT(0),
        β = FT(0),
    )

    model_2D = ShallowWaterModel{FT}(
        param_set,
        problem,
        ShallowWater.ConstantViscosity{FT}(model_3D.νʰ),
        nothing;
        coupling = coupling,
        c = FT(1),
        fₒ = FT(0),
        β = FT(0),
    )

    dt_fast = 300
    nout = ceil(Int64, tout / dt_slow)
    dt_slow = tout / nout

    vert_filter = CutoffFilter(grid_3D, polynomialorder(grid_3D) - 1)
    exp_filter = ExponentialFilter(grid_3D, 1, 8)

    integral_model = DGModel(
        VerticalIntegralModel(model_3D),
        grid_3D,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    modeldata = (
        vert_filter = vert_filter,
        exp_filter = exp_filter,
        integral_model = integral_model,
    )

    dg_3D = DGModel(
        model_3D,
        grid_3D,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        modeldata = modeldata,
    )

    dg_2D = DGModel(
        model_2D,
        grid_2D,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q_3D = init_ode_state(dg_3D, FT(0); init_on_cpu = true)
    Q_2D = init_ode_state(dg_2D, FT(0); init_on_cpu = true)

    lsrk_3D = LSRK54CarpenterKennedy(dg_3D, Q_3D, dt = dt_slow, t0 = 0)
    lsrk_2D = LSRK54CarpenterKennedy(dg_2D, Q_2D, dt = dt_fast, t0 = 0)
    odesolver = SplitExplicitSolver(lsrk_3D, lsrk_2D)

    step = [0, 0]
    cbvector = make_callbacks(
        vtkpath,
        step,
        nout,
        mpicomm,
        odesolver,
        dg_3D,
        model_3D,
        Q_3D,
        dg_2D,
        model_2D,
        Q_2D,
    )

    eng0 = norm(Q_3D)
    @info @sprintf """Starting
    norm(Q₀) = %.16e
    ArrayType = %s""" eng0 ArrayType

    # slow fast state tuple
    Qvec = (slow = Q_3D, fast = Q_2D)
    solve!(Qvec, odesolver; timeend = timeend, callbacks = cbvector)

    Qe_3D = init_ode_state(dg_3D, timeend, init_on_cpu = true)
    Qe_2D = init_ode_state(dg_2D, timeend, init_on_cpu = true)

    error_3D = euclidean_distance(Q_3D, Qe_3D) / norm(Qe_3D)
    error_2D = euclidean_distance(Q_2D, Qe_2D) / norm(Qe_2D)

    println("3D error = ", error_3D)
    println("2D error = ", error_2D)
    @test isapprox(error_3D, FT(0.0); atol = 0.005)
    @test isapprox(error_2D, FT(0.0); atol = 0.005)

    ## Check results against reference
    ClimateMachine.StateCheck.scprintref(cbvector[end])
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cbvector[end], refDat)
    end

    return nothing
end

function make_callbacks(
    vtkpath,
    step,
    nout,
    mpicomm,
    odesolver,
    dg_slow,
    model_slow,
    Q_slow,
    dg_fast,
    model_fast,
    Q_fast,
)
    if isdir(vtkpath)
        rm(vtkpath, recursive = true)
    end
    mkpath(vtkpath)
    mkpath(vtkpath * "/slow")
    mkpath(vtkpath * "/fast")

    function do_output(span, step, model, dg, Q)
        outprefix = @sprintf(
            "%s/%s/mpirank%04d_step%04d",
            vtkpath,
            span,
            MPI.Comm_rank(mpicomm),
            step
        )
        @info "doing VTK output" outprefix
        statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
        auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
        writevtk(outprefix, Q, dg, statenames, dg.state_auxiliary, auxnames)
    end

    do_output("slow", step[1], model_slow, dg_slow, Q_slow)
    cbvtk_slow = GenericCallbacks.EveryXSimulationSteps(nout) do (init = false)
        do_output("slow", step[1], model_slow, dg_slow, Q_slow)
        step[1] += 1
        nothing
    end

    do_output("fast", step[2], model_fast, dg_fast, Q_fast)
    cbvtk_fast = GenericCallbacks.EveryXSimulationSteps(nout) do (init = false)
        do_output("fast", step[2], model_fast, dg_fast, Q_fast)
        step[2] += 1
        nothing
    end

    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q_slow)
            @info @sprintf(
                """Update
                simtime = %8.2f / %8.2f
                runtime = %s
                norm(Q) = %.16e""",
                ODESolvers.gettime(odesolver),
                timeend,
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end

    cbcs_dg = ClimateMachine.StateCheck.sccreate(
        [
            (Q_slow, "3D state"),
            (dg_slow.state_auxiliary, "3D aux"),
            (Q_fast, "2D state"),
            (dg_fast.state_auxiliary, "2D aux"),
        ],
        nout;
        prec = 12,
    )

    return (cbvtk_slow, cbvtk_fast, cbinfo, cbcs_dg)
end