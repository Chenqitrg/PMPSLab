@testset "PeriodicTransferMatrix — construction from one MPS" begin
    N, d, D = 4, 2, 2
    mps = random_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)

    @test length(ptm) == N
    @test parent(ptm) === ptm.matrices
    @test eachindex(ptm) == eachindex(ptm.matrices)
    @test collect(ptm) == collect(ptm.matrices)
    for i in 1:N
        @test ptm[i] isa DenseTransferMatrix
        @test ptm[i] ≈ DenseTransferMatrix(mps[i])
    end
    new = DenseTransferMatrix(random_mps(ComplexF64, 1, d, D)[1])
    ptm[2] = new
    @test ptm[2] ≈ new
end

@testset "PeriodicTransferMatrix — construction from (bra, ket)" begin
    N, d, D = 3, 2, 3
    bra = random_mps(ComplexF64, N, d, D)
    ket = random_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(bra, ket)

    @test length(ptm) == N
    @test ptm[2] ≈ DenseTransferMatrix(bra[2], ket[2])
    @test edge_identity(ptm) ≈ id(ComplexF64, codomain(ptm[1]))
end

@testset "PeriodicTransferMatrix — mismatched periods throw" begin
    bra = random_mps(ComplexF64, 2, 2, 2)
    ket = random_mps(ComplexF64, 3, 2, 2)

    @test_throws DimensionMismatch PeriodicTransferMatrix(bra, ket)
    @test_throws DimensionMismatch dot(bra, ket)
    @test_throws DimensionMismatch isapprox(bra, ket)
end

@testset "PeriodicTransferMatrix — mixed 2-site/1-site construction" begin
    D, d = 2, 2
    mpsB = random_mps(ComplexF64, 4, d, D)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, d, D))
    ptm = PeriodicTransferMatrix(mpsA, mpsB)

    @test length(ptm) == 2
    @test ptm[1] ≈ DenseTransferMatrix(mpsA[1], mpsB[1], mpsB[2])
    @test ptm[2] ≈ DenseTransferMatrix(mpsA[2], mpsB[3], mpsB[4])

    mpsB_short = random_mps(ComplexF64, 3, d, D)
    @test_throws DimensionMismatch PeriodicTransferMatrix(mpsA, mpsB_short)
end

@testset "full-period product and expectation — 1-site" begin
    N, d, D = 3, 2, 2
    mps = random_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)

    E1, E2, E3 = ptm[1], ptm[2], ptm[3]
    @test *(ptm) ≈ E1 * E2 * E3
    @test expectation(ptm) ≈ overlap(mps, mps)

    ket = random_mps(ComplexF64, N, d, D)
    @test expectation(PeriodicTransferMatrix(mps, ket)) ≈ overlap(mps, ket)
end

@testset "full-period product and expectation — one_mps analytic" begin
    N, d, D = 4, 2, 2
    mps = one_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)
    @test expectation(ptm) ≈ d^N * D^(2N)
end

@testset "full-period product and expectation — 2-site MPS" begin
    N, d, D = 2, 2, 2
    mps = random_mps2(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)

    @test length(ptm) == N
    @test expectation(ptm) ≈ overlap(mps, mps)
    @test *(ptm) ≈ ptm[1] * ptm[2]
end

@testset "expectation — mixed 2-site bra / 1-site ket" begin
    D, d = 2, 2
    mpsB = random_mps(ComplexF64, 4, d, D)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, d, D))
    ptm = PeriodicTransferMatrix(mpsA, mpsB)

    @test expectation(ptm) ≈ overlap(mpsA, mpsB)
end

@testset "expectation — Z2-graded spaces" begin
    N = 3
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    bra = random_mps(ComplexF64, N, P, V)
    ket = random_mps(ComplexF64, N, P, V)

    @test expectation(PeriodicTransferMatrix(bra, ket)) ≈ overlap(bra, ket)
    @test expectation(PeriodicTransferMatrix(bra)) ≈ overlap(bra, bra)
end
