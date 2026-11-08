@testset "PeriodicTransferMatrix integration" begin
    N, d, D = 4, 2, 2
    mps = rand(PeriodicMPS, ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)
    v = build_cache(ptm)

    @test length(v) == N
    @test edge_space(v) == edge_space(ptm)
    @test edge_space(v) == codomain(ptm[1])

    # Every representable range matches a direct ordered product.
    @test v[1:1] ≈ range_product(ptm, 1:1)
    @test v[3:4] ≈ range_product(ptm, 3:4)
    @test v[4:4] ≈ range_product(ptm, 4:4)

    @test left_cache_before(v, 2) ≈ range_product(ptm, 1:1)
    @test right_cache_after(v, 2) ≈ range_product(ptm, 3:4)
    @test environment(v, 2) ≈ ptm[3] * ptm[4] * ptm[1]

    # The full-period helper agrees with a direct product.
    @test *(ptm) ≈ reduce(*, collect(ptm))
end
