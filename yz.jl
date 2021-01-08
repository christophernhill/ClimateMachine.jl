# Initialize
using ClimateMachine
using MPI
#
const FT = Float64
ClimateMachine.init()
ArrayType = ClimateMachine.array_type()
mpicomm = MPI.COMM_WORLD;

# Create a grid and save grid parameters
xOrderedEdgeList=[0,3]
# yOrderedEdgeList=[0,1,2,3,4,5,6,7,8,9,10]
# zOrderedEdgeList=[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0]
yOrderedEdgeList=[0,10]
# zOrderedEdgeList=[-10,0]
zOrderedEdgeList=[-3,-2,-1,0]

f(x)=0.5*(x[1]+x[end])
xmid=f(xOrderedEdgeList)
ymid=f(yOrderedEdgeList)
zmid=f(zOrderedEdgeList)
f(x)=x[end]-x[1]
Lx=f(xOrderedEdgeList)
Ly=f(yOrderedEdgeList)
Lz=f(zOrderedEdgeList)

using ClimateMachine.Mesh.Topologies
brickrange=( xOrderedEdgeList, yOrderedEdgeList, zOrderedEdgeList )
topl = StackedBrickTopology(
       mpicomm,
       brickrange;
#      periodicity=(true,true),
#      boundary=((0,0),(0,0)),
       periodicity=(true,false,false),
       boundary=((0,0),(1,1),(1,1)),
)

Np=4
using ClimateMachine.Mesh.Grids
mgrid = DiscontinuousSpectralElementGrid(
     topl,
     FloatType = FT,
     DeviceArray = ArrayType,
     polynomialorder = Np,
)

# Import an equation set template
include("test/Ocean/OcnCadj/OCNCADJEEquationSet.jl")
using ..OCNCADJEEquationSet

# Set up custom function and parameter options as needed
"""
  θ(t=0)=0
  θ(t=0)=exp(-((x-x0)/L0x)^2).exp(-((y-y0)/L0y)^2)
"""
const xDecayLength=FT(Lx/6)
const yDecayLength=FT(Ly/6)
const zDecayLength=FT(Lz/6)
function init_theta(x::FT,y::FT,z::FT,n,e)
 # xAmp=exp(  -( ( (x - xmid)/xDecayLength )^2 )  )
 # yAmp=exp(  -( ( (y - ymid)/yDecayLength )^2 )  )
 zAmp=exp(  -( ( (z - zmid)/zDecayLength )^2 )  )
 xAmp=1.
 yAmp=1.

 yAmp=1.
 xAmp=1. 
 zAmp=z+20   # Linear ramp + offset to make it more theta like!
 return FT(xAmp*yAmp*zAmp)
end

"""
  θˢᵒᵘʳᶜᵉ(1,1)=1
"""
function source_theta(θ,npt,elnum,x,y,z)
 if Int(npt) == 1 && Int(elnum) == 1
   return FT(0)
 end
 return FT(0)
end

"""
  Save array indexing and real world coords
"""
function init_aux_geom(npt,elnum,x,y,z)
 return npt,elnum,x,y,z
end

"""
  Compute and set diffusivity in each direction
"""
function calc_kappa_diff(∇θ,npt,elnum,x,y,z)
	return +0.1, +0.1, +0.1
  # return -0., -0., -0.
end

function get_wavespeed()
  return FT(0.)
end

const ptau = 1.
function get_penalty_tau()
  return FT(ptau)
end

# Add customizations to properties
bl_prop=OCNCADJEEquationSet.prop_defaults()
bl_prop=(bl_prop...,   init_aux_geom=init_aux_geom)
bl_prop=(bl_prop...,      init_theta=init_theta   )
bl_prop=(bl_prop...,    source_theta=source_theta )
bl_prop=(bl_prop..., calc_kappa_diff=calc_kappa_diff )
bl_prop=(bl_prop...,   get_wavespeed=get_wavespeed )
bl_prop=(bl_prop..., get_penalty_tau=get_penalty_tau )

# Create an equation set with the cutomized function and parameter properties
oml=OCNCADJEEquationSet.OCNCADJEEquations{Float64}(;bl_prop=bl_prop)

# Instantiate a DG model with the customized equation set
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
oml_dg = DGModel(oml,
                 mgrid,
                 RusanovNumericalFlux(),
                 CentralNumericalFluxSecondOrder(),
                 CentralNumericalFluxGradient(),
                 direction = VerticalDirection())
# oml_dg = DGModel(oml,mgrid,RusanovNumericalFlux(),PenaltyNumFluxDiffusive(),CentralNumericalFluxGradient())
oml_Q = init_ode_state(oml_dg, FT(0); init_on_cpu = true)
dQ = init_ode_state(oml_dg, FT(0); init_on_cpu = true)

# Execute the DG model
oml_dg(dQ,oml_Q, nothing, 0; increment=false)

println(oml_Q.θ)
println(dQ.θ)

# exit()

θ_0=deepcopy(oml_Q.θ)

M = view(mgrid.vgeo, :, mgrid.Mid, :)
println( sum(sum(M.*oml_Q.θ[:,1,:],dims=1)./sum(M , dims = 1)) )

# Make some plots
using Plots
using ClimateMachine.Mesh.Elements: interpolationmatrix

## Get some grid variables
dim = dimensionality(mgrid)
N = polynomialorders(mgrid)
Npx=(polynomialorders(mgrid)[1]+1)
Npy=(polynomialorders(mgrid)[1]+1)
Npz=(polynomialorders(mgrid)[1]+1)
Nq = N .+ 1
dxmin=minimum(mgrid.vgeo[2,13,:]-mgrid.vgeo[1,13,:])
dymin=minimum(mgrid.vgeo[Npx+1,14,:]-mgrid.vgeo[1,14,:])
dzmin=minimum(mgrid.vgeo[Npx*Npy+1,15,:]-mgrid.vgeo[1,15,:])
dlmin=minimum([dxmin,dymin,dzmin])

## Set interpolation
nsp=20
sp=range(-0.98;length=nsp,stop=0.98)
ξ = ntuple(
     i -> N[i] == 0 ? FT.([-1, 1]) : referencepoints(mgrid)[i],
     dim,
    )
ξdst=sp
I1d = ntuple(i -> interpolationmatrix(ξ[dim - i + 1], ξdst), dim)
I = kron(I1d...)

## Get interpolated grid locations
X=ntuple(i -> I*mgrid.x_vtk[i], length(mgrid.x_vtk) )

## Choose dt
dt_cfl_diff=dlmin^2/0.1*0.5
dt_penalty_diff=dlmin/ptau*0.5

dt=minimum([dt_cfl_diff,dt_penalty_diff])

## Set points to plot
fshp=ntuple(i -> nsp,dim,)
s1=Int(round(nsp/2))        # Coord 1 points
s2=Int(round(nsp/2))        # Coord 2 points
s3=1:nsp                    # Coord 3 points
ci=3                        # Coordinate index of the non-constant coordinate
xpts(i)=reshape(X[ci][:,i],fshp )[s1,s2,s3] # x-axis coordinates for non-constant coordinate

## 
mean_value_0=sum(sum(M.*oml_Q.θ[:,1,:],dims=1)./sum(M , dims = 1))
println("tic")
# anim = @animate
anim = @animate for iter=1:1000
 println(iter)
 oml_Q.θ.=oml_Q.θ-dt.*dQ.θ

 nelem = size(dQ)[end]
 # fld=dQ.θ
 fld=oml_Q.θ

 global fldsp=I*fld[:,1,:]
 global fldsp2=I*θ_0[:,1,:]
 i=1;plot( xpts(i) ,reshape(fldsp[ :,i],fshp )[s1,s2,s3] )
 i=2;plot!(xpts(i) ,reshape(fldsp[ :,i],fshp )[s1,s2,s3] )
 i=2;plot!(xpts(i) ,reshape(fldsp2[:,i],fshp )[s1,s2,s3] )
 i=3;plot!(xpts(i) ,reshape(fldsp[ :,i],fshp )[s1,s2,s3] )
 oml_dg(dQ,oml_Q, nothing, 0; increment=false);

end
println("toc")

println( sum(sum(M.*oml_Q.θ[:,1,:],dims=1)./sum(M , dims = 1)) - mean_value_0 )

gif(anim, "anim_fps15.gif", fps = 15)

## i=1;plot(reshape(X[1][:,i],(40,40))[:,20],reshape(fldsp[:,i],(40,40))[:,20] )
## i=2;plot!(reshape(X[1][:,i],(40,40))[:,20],reshape(fldsp[:,i],(40,40))[:,20] )
## i=3;plot!(reshape(X[1][:,i],(40,40))[:,20],reshape(fldsp[:,i],(40,40))[:,20] )


# Try some timestepping


