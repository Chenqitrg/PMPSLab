# Construct a periodic MPS
# Run from PMPSLab: julia --project=. examples/01_construction.jl
# This file runs independently; no other example needs to run first.

using PMPSLab
using Random: Xoshiro
using Test
using TensorKit

# An explicit generator makes this example reproducible; it is optional.
# You can also call rand(PeriodicMPS, ComplexF64, N, d, D).
rng = Xoshiro(20261102)

# Read: src/representation/constructors.jl and src/representation/container.jl.
# rand is extended by PMPSLab to construct N random site tensors.
# Each site maps right virtual space to (left virtual ⊗ physical) space.
# N: site count; d: physical dimension; D: virtual bond dimension.

N, d, D = 4, 2, 2
mps = rand(rng, PeriodicMPS, ComplexF64, N, d, D)
@show length(mps) codomain(mps[1]) domain(mps[1])

# A tuple requests multiple physical legs on each site.
two_leg = rand(rng, PeriodicMPS, ComplexF64, 2, (2, 3), D)
# one_mps creates all-ones site tensors; it does not normalize the state.
uniform = one_mps(ComplexF64, N, d, D)

@testset "Construction" begin
    @test length(mps) == N
    @test assertconsistency(mps) === nothing
    @test codomain(mps[1]) == ℂ^D ⊗ ℂ^d
    @test domain(mps[1]) == domain(uniform[1])
    @test codomain(two_leg[1]) == ℂ^D ⊗ ℂ^2 ⊗ ℂ^3
    @test size(convert(Array, dense(two_leg))) == (2, 3, 2, 3)
end
