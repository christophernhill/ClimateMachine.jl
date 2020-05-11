# # Heat equation tutorial

# In this tutorial, we'll be solving the [heat equation](https://en.wikipedia.org/wiki/Heat_equation):

# ``
# \frac{∂ T}{∂ t} + ∇ ⋅ (-α ∇T) = 0
# ``

# where
#  - `α` is the thermal diffusivity
#  - `T` is the temperature (the unknown we're solving for)

# To put this in the form of ClimateMachine's [`BalanceLaw`](@ref ClimateMachine.DGMethods.BalanceLaw), we'll re-write the equation as:

# ``
# \frac{∂ T}{∂ t} + ∇ ⋅ (F(T,t)) = 0
# ``

# where
#  - `F(T,t) = -α ∇T` is the diffusive flux

# with boundary conditions
#  - Fixed temperature `T_surface` at z_{min} (non-zero Dirichlet)
#  - No thermal flux at z_{min} (zero Neumann)

# Solving these equations is broken down into the following steps:
# - 1) Preliminary configuration
# - 2) PDEs
# - 3) Space discretization
# - 4) Time discretization
# - 5) Solver hooks / callbacks
# - 6) Solve
# - 7) Post-processing

# # Preliminary configuration

# ## Loading code

# First, we'll load our pre-requisites
#  - load external packages:
using MPI
using Test
using Logging
using Printf
using NCDatasets
using LinearAlgebra
using OrderedCollections
using Interpolations
using DelimitedFiles
using Plots
using StaticArrays

#  - load necessary ClimateMachine modules:
using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Writers
using ClimateMachine.VTK
using ClimateMachine.Mesh.Elements: interpolationmatrix
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.DGmethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates

#  - import necessary ClimateMachine modules: (`import` indicates that we must provide implementations of these structs/methods)
import ClimateMachine.DGmethods:
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    source!,
    flux_second_order!,
    flux_first_order!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    update_auxiliary_state!,
    nodal_update_auxiliary_state!,
    init_state_auxiliary!,
    init_state_conservative!,
    boundary_state!

# ## Initialization

# Define the float type (`Float64` or `Float32`)
FT = Float64;
# Initialize ClimateMachine for CPU.
ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
# Load some helper functions (soon to be incorporated into `ClimateMachine/src`)
include(joinpath(clima_dir, "tutorials", "Land", "helper_funcs.jl"));
include(joinpath(clima_dir, "tutorials", "Land", "plotting_funcs.jl"));

# # Define the set of Partial Differential Equations (PDEs)

# Model parameters can be stored in the particular [`BalanceLaw`](@ref ClimateMachine.DGMethods.BalanceLaw), in this case, a `HeatModel`:

Base.@kwdef struct HeatModel{FT} <: BalanceLaw
    "Heat capacity"
    ρc::FT = 1
    "Thermal diffusivity"
    α::FT = 0.01
    "Initial conditions for temperature"
    initialT::FT = 295.15
    "Surface boundary value for temperature (Dirichlet boundary conditions)"
    surfaceT::FT = 300.0
end

# Create an instance of the `HeatModel`:
m = HeatModel{FT}();

# All of the methods defined below, in this section, were `import`ed in [Loading code](@ref), and we must provide our own implementation as these methods are called inside `solve!`. Each of these methods have a fall-back of "do nothing" if we don't, or fail to, implement these methods after `import`ing.

# Specify auxiliary variables for `HeatModel` (stored in `aux`)
vars_state_auxiliary(::HeatModel, FT) = @vars(z::FT, T::FT);

# Specify state variables, the variables solved for in the PDEs, for `HeatModel` (stored in `Q`)
vars_state_conservative(::HeatModel, FT) = @vars(ρcT::FT);

# Specify state variables whose gradients are needed for `HeatModel`
vars_state_gradient(::HeatModel, FT) = @vars(T::FT);

# Specify gradient variables for `HeatModel`
vars_state_gradient_flux(::HeatModel, FT) = @vars(∇T::SVector{3, FT});

# Specify the initial values in `aux::Vars`, which are available in `init_state_conservative!`. Note that
# - this method is only called at `t=0`
# - `aux.z` and `aux.T` are available here because we've specified `z` and `T` in `vars_state_auxiliary`
function init_state_auxiliary!(m::HeatModel, aux::Vars, geom::LocalGeometry)
    aux.z = geom.coord[3]
    aux.T = m.initialT
end;

# Specify the initial values in `state::Vars`. Note that
# - this method is only called at `t=0`
# - `state.ρcT` is available here because we've specified `ρcT` in `vars_state_conservative`
function init_state_conservative!(
    m::HeatModel,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    state.ρcT = m.ρc * aux.T
end;

# The remaining methods, defined in this section, are called at every time-step in the solver by the [`BalanceLaw`](@ref ClimateMachine.DGMethods.BalanceLaw) framework.

# Overload `update_aux!` to call `soil_nodal_update_aux!`, or any other auxiliary methods
function update_aux!(
    dg::DGModel,
    m::HeatModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_aux!(soil_nodal_update_aux!, dg, m, Q, t, elems)
    return true # TODO: remove return true
end;

# Compute/update all auxiliary variables at each node. Note that
# - `aux.T` is available here because we've specified `T` in `vars_state_auxiliary`
function soil_nodal_update_aux!(m::HeatModel, state::Vars, aux::Vars, t::Real)
    aux.T = state.ρcT / m.ρc
end;

# Since we have diffusive fluxes, we must tell ClimateMachine to compute the gradient of `T`. Here, we specify how `T` is computed. Note that
#  - `transform.T` is available here because we've specified `T` in `vars_state_gradient`
function compute_gradient_argument!(
    m::HeatModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.T = state.ρcT / m.ρc
end;

# Specify where in `diffusive::Vars` to store the computed gradient in `compute_gradient_argument!`. Note that:
#  - `diffusive.∇T` is available here because we've specified `∇T` in `vars_state_gradient_flux`
#  - `∇transform.T` is available here because we've specified `T`  in `vars_state_gradient`
function compute_gradient_flux!(
    m::HeatModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.∇T = ∇transform.T
end;

# We do no have sources, nor non-diffusive fluxes.
function source!(m::HeatModel, _...) end;
function flux_first_order!(
    m::HeatModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end;

# Compute diffusive flux (``F(T,t) = -α ∇T`` in the original PDE). Note that:
# - `diffusive.∇T` is available here because we've specified `∇T` in `vars_state_gradient_flux`
function flux_second_order!(
    m::HeatModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρcT -= m.α * diffusive.∇T
end;

# ### Boundary conditions

# Second-order terms in our equations, ``∇⋅(αG)`` where ``G=∇T``, are internally reformulated to first-order unknowns.
# Boundary conditions must be specified for all unknowns, both first-order and second-order unknowns which have been reformulated.

# The boundary conditions for `ρcT` are specified here for first-order unknowns
function boundary_state!(
    nf,
    m::HeatModel,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    if bctype == 1 # surface
        state⁺.ρcT = m.ρc * m.surfaceT
    elseif bctype == 2 # bottom
        nothing
    end
end;

# The boundary conditions for `ρcT` are specified here for second-order unknowns
function boundary_state!(
    nf,
    m::HeatModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    if bctype == 1 # surface
        state⁺.ρcT = m.ρc * m.surfaceT
    elseif bctype == 2 # bottom
        diff⁺.∇T = -diff⁻.∇T
    end
end;

# # Spatial discretization

# Prescribe vector of vertical elements (in meters)
velems = collect(0:10) / 10;

# Prescribe polynomial order of basis functions in finite elements
N_poly = 5;

# Define a grid
grid = SingleStackGrid(MPI, velems, N_poly, FT, Array);

# Configure the DG model, based on the PDEs, grid and penalty terms
dg = DGModel(
    m,
    grid,
    CentralNumericalFluxFirstOrder(),
    CentralNumericalFluxSecondOrder(),
    CentralNumericalFluxGradient(),
);

# # Time discretization

# We'll define the time-step based on the [Fourier number](https://en.wikipedia.org/wiki/Fourier_number)
Δ = min_node_distance(grid)

given_Fourier = 0.08;
Fourier_bound = given_Fourier * Δ^2 / m.α;
dt = Fourier_bound

# ## Initialize the state vector
# This initializes the state vector and allocates memory for the solution in space (`dg` has the model `m`, which describes the PDEs as well as the function used for initialization):
Q = init_ode_state(dg, FT(0));

# ## Initialize the ODE solver
# Here, we use an explicit Low-Storage [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) method
lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0);

# ## Inspect the initial conditions

# Let's export a plot of the initial state
output_dir = @__DIR__;

mkpath(output_dir);

state_vars = get_vars_from_stack(grid, Q, m, vars_state_conservative);
aux_vars =
    get_vars_from_stack(grid, dg.state_auxiliary, m, vars_state_auxiliary);
all_vars = OrderedDict(state_vars..., aux_vars...);
export_plot_snapshot(
    all_vars,
    ("ρcT",),
    joinpath(output_dir, "initial_condition.png"),
);
# ![](initial_condition.png)

# It matches what we have in `init_state_conservative!(m::HeatModel, ...)`, so let's continue.

# # Solver hooks / callbacks

# Define simulation time (all units are SI) and number of outputs over this interval
const timeend = 40;
const n_outputs = 5;

# This equates to exports every ceil(Int, timeend/n_outputs) time-step:
const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# Create a dictionary for `z` coordinate (and convert to cm) NCDatasets IO:
z_scale = 100 # convert from meters to cm
dims = OrderedDict("z" => collect(get_z(grid, z_scale)));

# Create a DataFile, which is callable to get the name of each file given a step
output_data = DataFile(joinpath(output_dir, "output_data"));

# ClimateMachine's time-steppers provide hooks, or callbacks, which allow users to inject code to be executed at specified intervals. In this callback, the state and aux variables are collected, combined into a single `OrderedDict` and written to a NetCDF file (for each output step `step`).
step = [0];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
    lsrk,
) do (init = false)
    state_vars = get_vars_from_stack(
        grid,
        Q,
        m,
        vars_state_conservative;
        exclude = [],
    )
    aux_vars = get_vars_from_stack(
        grid,
        dg.state_auxiliary,
        m,
        vars_state_auxiliary;
        exclude = ["z"],
    )
    all_vars = OrderedDict(state_vars..., aux_vars...)
    write_data(
        NetCDFWriter(),
        output_data(step[1]),
        dims,
        all_vars,
        gettime(lsrk),
    )
    step[1] += 1
    nothing
end;

# # Solve

# This is the main "solve" method call. While users do not have access to the time-stepping loop, code and function calls may be injected via `callbacks` keyword argument, which is a `Tuple` of [`GenericCallbacks`](@ref).
solve!(Q, lsrk; timeend = timeend, callbacks = (callback,));

# # Post-processing

# Our solution has now been calculated and exported to NetCDF files in `output_dir`. Let's collect them all into a dictionary whose keys are output interval and values of NCDataset's:

all_data = collect_data(output_data, step[1]);

# To get `T` at ``t=0``, we can use `T_at_t_0 = all_data[0]["T"][:]`

# Let's plot the solution:

export_plot(all_data, ("ρcT",), joinpath(output_dir, "solution_vs_time.png"));
# ![](solution_vs_time.png)

# The results look as we would expect: a fixed temperature at the bottom is resulting in heat flux that propagates up the domain. To run this file, and inspect the solution in `all_data`, include this tutorial in the Julia REPL with:

# ```julia
# include(joinpath("tutorials", "Land", "Heat", "heat_equation.jl"))
# ```
