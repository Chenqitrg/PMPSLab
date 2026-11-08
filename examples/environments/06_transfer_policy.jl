# Run from PMPSLab: julia --project=. examples/environments/06_transfer_policy.jl
# A policy selects the representation; a transfer matrix holds numerical data.
using PMPSLab: PeriodicMPS
using PMPSLab: DenseTransferPolicy, TruncatedTransferPolicy, transfer_at, build_transfer_cache
using PeriodicProductsLab: environment
using TensorKit: TensorMap, truncrank, space, dim
using LinearAlgebra: norm
using Test: @test

mps = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
exact_policy = DenseTransferPolicy()
compressed_policy = TruncatedTransferPolicy(; trunc=truncrank(1))

exact = transfer_at(mps, 2, exact_policy)
compressed = transfer_at(mps, 2, compressed_policy)
@test dim(space(compressed.s, 1)) == 1
@show norm(TensorMap(exact) - TensorMap(compressed)) / norm(TensorMap(exact))

# The cache uses the same policy, truncating again as products are accumulated.
cache = build_transfer_cache(mps, mps, compressed_policy)
local_environment = environment(cache, 2)
@test dim(space(local_environment.s, 1)) == 1
@show space(local_environment.s, 1)
# Truncated environments are approximate and need not equal dense environments.
# Build a new cache when switching policies; no exact cache is reused here.
nothing
