# Run from PMPSLab: julia --project=. examples/environments/03_overlap.jl
# The overlap <bra|ket> closes the virtual legs of the full-period transfer.
using PMPSLab: one_mps, dense
using PMPSLab: expectation, PeriodicTransferMatrix
using TensorKit: TensorMap
using LinearAlgebra: dot, tr
using Test: @test

# Three two-level sites, bond dimension 1; every physical amplitude starts at 1.
bra = one_mps(ComplexF64, 3, 2, 1)
ket = one_mps(ComplexF64, 3, 2, 1)
ket[2] = (1 + im) * ket[2]

value = expectation(bra, ket)
@test value ≈ 8 * (1 + im)
@test expectation(ket, bra) ≈ conj(value)
@test dot(bra, ket) ≈ value

# Two references: trace the transfer product, or take the full-state inner product.
transfers = PeriodicTransferMatrix(bra, ket)
@test value ≈ tr(TensorMap(*(transfers)))
@test value ≈ dot(dense(bra), dense(ket))
@show value
# Here expectation means overlap; this call inserts no observable.
nothing
