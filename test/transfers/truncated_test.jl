using PMPSLab: DenseTransferMatrix, TruncatedTransferMatrix
@testset "TruncatedTransferMatrix — factored representation" begin
    D, d = 3, 2
    tensor = randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)
    transfer = DenseTransferMatrix(tensor)

    exact = TruncatedTransferMatrix(transfer, TruncationStrategy())
    @test exact isa AbstractTransferMatrix
    @test exact.trunc == TruncationStrategy()
    @test numout(exact.u) == 2
    @test numin(exact.u) == 1
    @test numout(exact.s) == 1
    @test numin(exact.s) == 1
    @test numout(exact.vᴴ) == 1
    @test numin(exact.vᴴ) == 2
    @test codomain(exact) == codomain(transfer)
    @test domain(exact) == domain(transfer)
    @test TensorMap(exact) ≈ TensorMap(transfer)
    @test TensorMap(DenseTransferMatrix(exact)) ≈ TensorMap(exact)
    @test TensorMap(TruncatedTransferMatrix(tensor, TruncationStrategy())) ≈
          TensorMap(transfer)

    trunc = truncrank(2)
    rank_two = TruncatedTransferMatrix(transfer, trunc)
    @test rank_two.trunc === trunc
    @test dim(space(rank_two.s, 1)) == 2
    @test norm(TensorMap(transfer) - TensorMap(rank_two)) < norm(TensorMap(transfer))
end

@testset "combine_truncation — discard permission is additive" begin
    rank = combine_truncation(truncrank(5), truncrank(3))
    @test rank isa MatrixAlgebraKit.TruncationByOrder
    @test rank.howmany == 3

    error = combine_truncation(
        truncerror(atol = 1e-12, rtol = 1e-10),
        truncerror(atol = 1e-9, rtol = 1e-13),
    )
    @test error isa MatrixAlgebraKit.TruncationByError
    @test error.atol == 1e-9
    @test error.rtol == 1e-10

    unchanged = truncerror(rtol = 1e-8)
    @test combine_truncation(unchanged, unchanged) === unchanged
    @test combine_truncation(TruncationStrategy(), unchanged) === unchanged

    mixed = combine_truncation(truncrank(3), truncerror(rtol = 1e-8))
    @test mixed isa MatrixAlgebraKit.TruncationIntersection
end

@testset "TruncatedTransferMatrix — factored multiplication" begin
    D, d = 3, 2
    dense_left = DenseTransferMatrix(randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D))
    dense_right = DenseTransferMatrix(randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D))

    left_exact = TruncatedTransferMatrix(dense_left, TruncationStrategy())
    right_exact = TruncatedTransferMatrix(dense_right, TruncationStrategy())
    exact_product = left_exact * right_exact
    @test exact_product isa TruncatedTransferMatrix
    @test TensorMap(exact_product) ≈ TensorMap(dense_left * dense_right)

    left = TruncatedTransferMatrix(dense_left, truncrank(5))
    right = TruncatedTransferMatrix(dense_right, truncrank(3))
    product = left * right
    @test product.trunc isa MatrixAlgebraKit.TruncationByOrder
    @test product.trunc.howmany == 3
    @test dim(space(product.s, 1)) == 3

    materialized_inputs = TensorMap(left) * TensorMap(right)
    expected_u, expected_s, expected_vᴴ, _ =
        svd_trunc(materialized_inputs; trunc = truncrank(3))
    expected = expected_u * expected_s * expected_vᴴ
    @test TensorMap(product) ≈ expected
end

@testset "TruncatedTransferMatrix — mul! cache reuse" begin
    D, d = 3, 2
    trunc = truncrank(3)
    left = TruncatedTransferMatrix(
        DenseTransferMatrix(randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)), trunc
    )
    right = TruncatedTransferMatrix(
        DenseTransferMatrix(randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)), trunc
    )
    destination = left * right
    u_slot, s_slot, vᴴ_slot = destination.u, destination.s, destination.vᴴ

    updated = mul!(destination, left, right)
    @test updated === destination
    @test updated.u === u_slot
    @test updated.s === s_slot
    @test updated.vᴴ === vᴴ_slot
    @test TensorMap(updated) ≈ TensorMap(left * right)

    narrower = TruncatedTransferMatrix(
        DenseTransferMatrix(left), truncrank(2)
    ) * TruncatedTransferMatrix(DenseTransferMatrix(right), truncrank(2))
    replacement = mul!(narrower, left, right)
    @test replacement !== narrower
    @test dim(space(replacement.s, 1)) == 3
    @test TensorMap(replacement) ≈ TensorMap(left * right)
end

@testset "TruncatedTransferMatrix — copy! cache reuse" begin
    D, d = 3, 2
    trunc = truncrank(3)
    source = TruncatedTransferMatrix(
        DenseTransferMatrix(randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)), trunc
    )
    destination = copy(source)
    u_slot, s_slot, vᴴ_slot = destination.u, destination.s, destination.vᴴ

    updated = copy!(destination, source)
    @test updated === destination
    @test updated.u === u_slot
    @test updated.s === s_slot
    @test updated.vᴴ === vᴴ_slot
    @test TensorMap(updated) ≈ TensorMap(source)

    narrower = TruncatedTransferMatrix(
        DenseTransferMatrix(randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)),
        truncrank(2),
    )
    replacement = copy!(narrower, source)
    @test replacement !== narrower
    @test replacement.trunc === trunc
    @test TensorMap(replacement) ≈ TensorMap(source)
end
