# Periodic MPS with Z₂ symmetry
# Run from PMPSLab: julia --project=. examples/09_symmetry.jl
# This file runs independently; no other example needs to run first.

using PMPSLab
using Random: Xoshiro
using Test
using TensorKit

# An explicit generator makes this example reproducible; it is optional.
# You can also call rand(PeriodicMPS, ComplexF64, N, d, D).
rng = Xoshiro(20261102)

# Read: test/contract_pairs_test.jl for broader pairing cases.
# TensorKit supplies sector-aware spaces; PMPSLab preserves them.
# The plain-array reference in 03_dense_contraction.jl uses ordinary complex spaces.

V = Z2Space(0 => 2, 1 => 1)
P = Z2Space(0 => 2)
symmetric_mps = rand(rng, PeriodicMPS, ComplexF64, 4, P, V)
println("\n9. A periodic MPS with Z₂-graded virtual bonds")
@show codomain(symmetric_mps[1])

@testset "Symmetry-aware pairing" begin
    @test assertconsistency(symmetric_mps) === nothing
    @test codomain(contract_pairs(symmetric_mps)[1]) == V ⊗ P ⊗ P
    @test dense(contract_pairs(symmetric_mps)) ≈ dense(symmetric_mps)
end
