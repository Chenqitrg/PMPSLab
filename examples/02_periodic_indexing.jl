# Periodic site and bond indexing
# Run from PMPSLab: julia --project=. examples/02_periodic_indexing.jl
# This file runs independently; no other example needs to run first.

using PMPSLab
using Random: Xoshiro
using Test
using TensorKit
using MPSKit: left_virtualspace, right_virtualspace

# An explicit generator makes this example reproducible; it is optional.
# You can also call rand(PeriodicMPS, ComplexF64, N, d, D).
rng = Xoshiro(20261102)

# Read: src/constructors.jl and src/container.jl
# A site tensor has the shape (left virtual ⊗ physical) ← right virtual.
# Integer positions select tensors. Query bond spaces using MPSKit functions:
# right_virtualspace(mps, i) is the space between sites i and i+1.
# Constructor input V[i] specifies this right space; site i has left space V[i-1].

N = 4
P = fill(ℂ^2, N)
V = [ℂ^2, ℂ^3, ℂ^4, ℂ^3]
mps = rand(rng, PeriodicMPS, ComplexF64, P, V)
println("\n2. Four sites with nonuniform right virtual spaces")
@show length(mps) codomain(mps[1]) domain(mps[1]) right_virtualspace(mps, 1)

@testset "Construction and periodic indexing" begin
    @test assertconsistency(mps) === nothing
    @test length(mps) == N
    @test mps[N + 1] === mps[1]
    @test mps[0] === mps[N]
    @test collect(right_virtualspace(mps)) == V
    @test collect(left_virtualspace(mps)) == circshift(V, 1)
    @test right_virtualspace(mps, 1) == left_virtualspace(mps, 2)
    @test right_virtualspace(mps, N + 1) == right_virtualspace(mps, 1)
    @test all(right_virtualspace(mps, i) == left_virtualspace(mps, i + 1)
              for i in 1:N)
end
