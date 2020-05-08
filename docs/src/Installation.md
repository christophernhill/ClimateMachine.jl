# Installing ClimateMachine

### Install Julia

The current release of `ClimateMachine` is verified to work with
Julia 1.3.1. Download it for your platform from [Julia's old
releases](https://julialang.org/downloads/oldreleases/#v131_dec_30_2019).

### Install MPI

If you're running on a cluster, MPI is likely installed -- check with your
IT staff.

Otherwise, download and install one of the following MPI implementations
for your platform:

- Windows -- [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi).
- MacOS -- [Open MPI](https://www.open-mpi.org/) or
[MPICH](https://www.mpich.org/), available on [Homebrew](https://brew.sh/)
and [MacPorts](https://www.macports.org/)
- Linux -- [OpenMPI](https://www.open-mpi.org/) or
[MPICH](https://www.mpich.org/), available in package managers

Then add [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) to Julia
using the built-in package manager (press `]` at the Julia prompt):

```julia
julia> ]
(v1.3) pkg> add MPI
```

The package should be installed and built without errors. You can verify
that all is well with:

```julia
julia> ]
(v1.3) pkg> test MPI
```

If you are having problems, see the
[`MPI.jl` documentation](https://juliaparallel.github.io/MPI.jl/stable/configuration.html)
for help.

### Install `ClimateMachine`

Download the `ClimateMachine` [source](https://github.com/CliMA/ClimateMachine.jl.git)
(you will need [`Git`](https://git-scm.com/):

```
git clone https://github.com/CliMA/ClimateMachine.jl.git
```

Install all the Julia packages required by `ClimateMachine` by running
```
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
```
from the `ClimateMachine.jl` directory.

You can verify your `ClimateMachine` installation by running
```
julia --project test/runtests.jl
```
from the `ClimateMachine.jl` directory. This will take a while!

You are now ready to run one of the tutorials. For instance, the dry
Rayleigh Benard tutorial:
```
julia --project tutorials/Atmos/dry_rayleigh_benard.jl
```

`ClimateMachine` is CUDA-enabled and will use GPU(s) if available. To
run on the CPU, set the environment variable `CLIMATEMACHINE_GPU` to
`false`.
