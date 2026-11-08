using PMPSLab: find_projectors
# Run from PMPSLab: julia --project=. examples/gauge/06_structural_cleanup.jl
# Advanced: repair virtual dimensions that prevent fixed-workspace canonicalization.
using PMPSLab: ProjectorGaugeFix, find_projectors, gaugefix!
using PMPSLab: PeriodicMPS, dense, assertconsistency
using MPSKit: right_virtualspace
using TensorKit: ℂ, ⊗, ←, dim
using Test: @test, @test_throws

# Alternating right bond dimensions 1, 3, 1, 3 with physical dimension 2.
# A dimension-3 bond adjacent to a dimension-1 bond cannot have full local rank.
mps = PeriodicMPS([
    rand(ComplexF64, ℂ^3 ⊗ ℂ^2 ← ℂ^1),
    rand(ComplexF64, ℂ^1 ⊗ ℂ^2 ← ℂ^3),
    rand(ComplexF64, ℂ^3 ⊗ ℂ^2 ← ℂ^1),
    rand(ComplexF64, ℂ^1 ⊗ ℂ^2 ← ℂ^3),
])
before = dense(mps)
algorithm = ProjectorGaugeFix()

# find_projectors currently requires dimensions compatible with both canonicalizations.
# Its fixed-workspace solver does not perform structural cleanup.
@test_throws ArgumentError find_projectors(mps, algorithm)
# gaugefix! first performs exact structural cleanup, then constructs projectors.
gaugefix!(mps, algorithm)
@test assertconsistency(mps) === nothing
@test [dim(right_virtualspace(mps, i)) for i in eachindex(mps)] == [1, 2, 1, 2]
@test dense(mps) ≈ before rtol=1e-9
@show [dim(right_virtualspace(mps, i)) for i in eachindex(mps)]
nothing
