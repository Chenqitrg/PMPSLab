using PMPSLab: DenseTransferMatrix, TruncatedTransferMatrix
# SVD factors and an explicit limit on the retained dimension.
using TensorKit: TensorMap, ℂ, ⊗, truncrank, space, dim
using LinearAlgebra: norm
using MPSKit: TruncationStrategy
using Test: @test

V, P = ℂ^2, ℂ^2
A = TensorMap(reshape(collect(1.0:8.0), 2, 2, 2), V ⊗ P, V)
E = DenseTransferMatrix(A)
exact_factors = TruncatedTransferMatrix(E, TruncationStrategy())
rank_one = TruncatedTransferMatrix(E, truncrank(1))

# Without truncation, u * s * vᴴ reconstructs the original transfer.
@test TensorMap(exact_factors) ≈ TensorMap(E)
@test TensorMap(exact_factors) ≈ exact_factors.u * exact_factors.s * exact_factors.vᴴ
@test dim(space(rank_one.s, 1)) == 1
@show space(exact_factors.s, 1) space(rank_one.s, 1)
@show norm(TensorMap(E) - TensorMap(rank_one)) / norm(TensorMap(E))

# Factored transfers can also be multiplied; the product retains a truncation policy.
product = rank_one * rank_one
@test dim(space(product.s, 1)) == 1
nothing
