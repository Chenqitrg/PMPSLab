# Rejected operations and bond invariants
# Run from PMPSLab: julia --project=. examples/06_validation.jl
# This file runs independently; no other example needs to run first.

using PMPSLab
using Random: Xoshiro
using Test
using TensorKit

# An explicit generator makes this example reproducible; it is optional.
# You can also call rand(PeriodicMPS, ComplexF64, N, d, D).
rng = Xoshiro(20261102)

N, d, D = 4, 2, 2
mps = rand(rng, PeriodicMPS, ComplexF64, N, d, D)
state = dense(mps)

# Read: src/container.jl, setindex!; src/contraction.jl, contract_pairs
# A replacement site must match both neighboring virtual spaces.
# @test_throws checks expected errors, allowing the walkthrough to continue.

println("\n6. Invalid bond dimensions and odd-length pairing are rejected")
@testset "Validation boundaries" begin
    candidate = deepcopy(mps)
    incompatible = rand(rng, ComplexF64, ℂ^3 ⊗ ℂ^d ← ℂ^D)
    @test_throws SpaceMismatch (candidate[1] = incompatible)
    @test dense(candidate) ≈ state
    odd_chain = rand(rng, PeriodicMPS, ComplexF64, 3, d, D)
    @test_throws ArgumentError contract_pairs(odd_chain)
    @test_throws MethodError mps[3 // 2]  # Tensor access uses integer sites only.
end
