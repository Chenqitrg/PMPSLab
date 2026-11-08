# Apply a function to every tensor
# Run from PMPSLab: julia --project=. examples/07_tensor_mapping.jl
# This file runs independently; no other example needs to run first.

using PMPSLab
using Random: Xoshiro
using Test

# An explicit generator makes this example reproducible; it is optional.
# You can also call rand(PeriodicMPS, ComplexF64, N, d, D).
rng = Xoshiro(20261102)

N, d, D = 4, 2, 2
mps = rand(rng, PeriodicMPS, ComplexF64, N, d, D)

# Read: src/representation/container.jl, map_tensors.
# Scaling each of N tensors by α scales the dense state by α^N.
# The original tensors are unchanged for this out-of-place scaling function.
# For an independent tensor copy, use map_tensors(copy, mps).
# map_tensors(identity, mps) retains aliases to the original tensor objects.

state = dense(mps)
scaled = map_tensors(tensor -> 2 * tensor, mps)
@show length(scaled)

@testset "Tensor mapping" begin
    @test dense(scaled) ≈ 2^N * state
    @test dense(mps) ≈ state
end
