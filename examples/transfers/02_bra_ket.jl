using PMPSLab: DenseTransferMatrix
# Two site tensors: the first is bra, the second is ket.
using TensorKit: TensorMap, ℂ, ⊗
using Test: @test

V, P = ℂ^2, ℂ^2
A = TensorMap(reshape(ComplexF64.(1:8), 2, 2, 2), V ⊗ P, V)
bra = (1 + im) * A
ket = A

self_transfer = DenseTransferMatrix(A)
mixed_transfer = DenseTransferMatrix(bra, ket)

# The bra is conjugated automatically, so its coefficient becomes 1 - im.
@test TensorMap(mixed_transfer) ≈ (1 - im) * TensorMap(self_transfer)
display(TensorMap(mixed_transfer))
nothing
