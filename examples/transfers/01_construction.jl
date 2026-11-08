using PMPSLab: DenseTransferMatrix
# Construct an exact transfer matrix from one site tensor.
using TensorKit: TensorMap, ℂ, ⊗, domain, codomain
using Test: @test

V, P = ℂ^2, ℂ^2
A = TensorMap(reshape(collect(1.0:8.0), 2, 2, 2), V ⊗ P, V)
E = DenseTransferMatrix(A)

# Contract the physical legs, retaining the left/right virtual legs of ket and bra.
@test domain(E) == V ⊗ V'
@test codomain(E) == V ⊗ V'
@show domain(E) codomain(E)
display(E)

# E wraps the transfer; TensorMap(E) returns the stored TensorKit data without copying.
@test TensorMap(E) === E.data
display(TensorMap(E))
nothing
