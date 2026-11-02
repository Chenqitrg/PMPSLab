# Full-period contraction
# Run from PMPSLab: julia --project=. examples/03_dense_contraction.jl
# This file runs independently; no other example needs to run first.

using PMPSLab
using Random: Xoshiro
using Test
using TensorKit
using LinearAlgebra: tr

# An explicit generator makes this example reproducible; it is optional.
# You can also call rand(PeriodicMPS, ComplexF64, N, d, D).
rng = Xoshiro(20261102)

N, d, D = 4, 2, 2
mps = rand(rng, PeriodicMPS, ComplexF64, N, d, D)

# Read: src/contraction.jl, dense
# For ordinary complex spaces, fixing each physical index gives a D×D matrix.
# The corresponding wavefunction amplitude is tr(A₁[p₁] ⋯ Aₙ[pₙ]).
# This reference uses ordinary arrays, rather than PMPSLab contractions.

state = dense(mps)
state_array = convert(Array, state)
site_arrays = [convert(Array, tensor) for tensor in mps]
println("\n3. Dense state retains every physical leg and traces out virtual legs")
@show size(state_array)

@testset "Dense contraction agrees with matrix traces" begin
    @test size(state_array) == (d, d, d, d)
    for indices in CartesianIndices(state_array)
        p = Tuple(indices)
        matrices = [site_arrays[i][:, p[i], :] for i in 1:N]
        @test state_array[indices] ≈ tr(reduce(*, matrices))
    end
    # one_mps means all-ones site tensors, not a normalized quantum state.
    # Each of the N closed virtual indices has D possible values.
    uniform = one_mps(ComplexF64, N, d, D)
    @test convert(Array, dense(uniform)) ≈ fill(ComplexF64(D^N), d, d, d, d)
end
