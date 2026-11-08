# Run from PMPSLab: julia --project=. examples/gauge/05_bond_compression.jl
# A truncation policy limits the retained virtual spaces and can change the state.
using PMPSLab: ProjectorGaugeFix, gaugefix!
using PMPSLab: PeriodicMPS, dense, assertconsistency
using MPSKit: right_virtualspace
using TensorKit: truncrank, dim
using LinearAlgebra: norm
using Test: @test

mps = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
before = dense(mps)
@show [dim(right_virtualspace(mps, i)) for i in eachindex(mps)]

algorithm = ProjectorGaugeFix(; trunc=truncrank(1))
gaugefix!(mps, algorithm)
@test assertconsistency(mps) === nothing
@test all(dim(right_virtualspace(mps, i)) == 1 for i in eachindex(mps))
@show [dim(right_virtualspace(mps, i)) for i in eachindex(mps)]
@show norm(dense(mps) - before) / norm(before)

# No state-equality assertion here: discarded virtual components can carry weight.
# A local rank limit does not specify a global state-error tolerance.
nothing
