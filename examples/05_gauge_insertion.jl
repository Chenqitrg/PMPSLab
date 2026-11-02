# Insert a cancelling gauge pair
# Run from PMPSLab: julia --project=. examples/05_gauge_insertion.jl
# This file runs independently; no other example needs to run first.

using PMPSLab
using Random: Xoshiro
using Test
using TensorKit
using MPSKit: right_virtualspace

# An explicit generator makes this example reproducible; it is optional.
# You can also call rand(PeriodicMPS, ComplexF64, N, d, D).
rng = Xoshiro(20261102)

N, d, D = 4, 2, 2
mps = rand(rng, PeriodicMPS, ComplexF64, N, d, D)
state = dense(mps)

# Read: src/container.jl, insert_gauge!, mul_left_virtual, mul_right_virtual
# The left tensor absorbs U; the right tensor absorbs U†. Since U U† = I,
# their contraction is unchanged. Use deepcopy before a mutation experiment.
# General insert_gauge! inputs need not be inverses: arbitrary projectors can
# change the represented state. Preservation here relies on our unitary pair.

gauged = deepcopy(mps)
site = 1  # Insert on the right of site 1, between sites 1 and 2.
bond_space = right_virtualspace(gauged, site)
U = unitary(rand(rng, ComplexF64, bond_space ← bond_space))
insert_gauge!(gauged, site, U, adjoint(U))
println("\n5. Gauge insertion changes local tensors but preserves the dense state")

@testset "Gauge cancellation" begin
    @test gauged[1] ≈ mul_right_virtual(mps[1], U)
    @test gauged[2] ≈ mul_left_virtual(adjoint(U), mps[2])
    @test gauged[3] == mps[3]
    @test gauged[4] == mps[4]
    @test assertconsistency(gauged) === nothing
    @test dense(gauged) ≈ state
    @test dense(mps) ≈ state
end
