# Contract adjacent pairs
# Run from PMPSLab: julia --project=. examples/04_pair_contraction.jl
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
state_array = convert(Array, state)

# Read: src/contraction.jl, contract_pairs
# Four one-physical-leg tensors become two two-physical-leg tensors.
# No truncation or optimization occurs here.

paired = contract_pairs(mps)
shifted_pairs = contract_pairs(mps, 2)
println("\n4. Pairing changes the grouping of tensors")
@show length(paired) codomain(paired[1]) domain(paired[1])

@testset "Pair contraction and physical ordering" begin
    @test length(paired) == N ÷ 2
    @test codomain(paired[1]) == ℂ^D ⊗ ℂ^d ⊗ ℂ^d
    @test dense(paired) ≈ state
    # Starting at 2 pairs (2,3) and (4,1); physical order becomes (2,3,4,1).
    @test dense(shifted_pairs) ≈ dense(circshift(mps, -1))
    @test convert(Array, dense(shifted_pairs)) ≈
          permutedims(state_array, (2, 3, 4, 1))
end
