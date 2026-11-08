using PMPSLab: DenseTransferMatrix
# Transfer composition and application order; no PeriodicMPS or cache is needed.
using TensorKit: TensorMap, ℂ, ⊗
using Test: @test

V, P = ℂ^2, ℂ^2
A = TensorMap(reshape(collect(1.0:8.0), 2, 2, 2), V ⊗ P, V)
B = TensorMap(reshape(collect(8.0:-1.0:1.0), 2, 2, 2), V ⊗ P, V)
E1, E2 = DenseTransferMatrix(A), DenseTransferMatrix(B)
X = TensorMap([1.0 2.0; 3.0 4.0], V, V)

# E1 * E2 applies E2 first, then E1, just like ordinary matrix multiplication.
combined = E1 * E2
@test TensorMap(combined) ≈ TensorMap(E1) * TensorMap(E2)
@test combined * X ≈ E1 * (E2 * X)
display(combined * X)
nothing
