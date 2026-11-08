# Repeat a finite unit cell
# Run from PMPSLab: julia --project=. examples/08_finite_lift.jl
# This file runs independently; no other example needs to run first.

using PMPSLab
using Random: Xoshiro
using Test

# An explicit generator makes this example reproducible; it is optional.
# You can also call rand(PeriodicMPS, ComplexF64, N, d, D).
rng = Xoshiro(20261102)

N, d, D = 4, 2, 2
mps = rand(rng, PeriodicMPS, ComplexF64, N, d, D)

# Read: src/representation/lift.jl.
# lift(mps, 2) concatenates two copies of the tensor sequence into a larger ring.
# This is a finite lift. It does not convert the state to an infinite MPS.
# Tensor objects may be shared; use deepcopy for independent mutation experiments.

repeated = lift(mps, 2)
@show length(mps) length(repeated)

@testset "Finite lifting" begin
    @test length(repeated) == 2N
    @test all(repeated[i] == mps[i] for i in 1:2N)
    @test assertconsistency(repeated) === nothing
end
