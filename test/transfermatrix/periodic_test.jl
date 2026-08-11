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
        @test transfer_at(mps, i) ≈ ptm[i]
        @test transfer_at(mps, i, DenseTransferPolicy()) ≈ ptm[i]
        @test transfer_at(mps, mps, i) ≈ ptm[i]
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
    @test transfer_at(bra, ket, 2) ≈ ptm[2]
    @test edge_identity(ptm) ==
          FormalIdentity{eltype(ptm)}(codomain(ptm[1]))

    direct = build_cache(bra, ket)
    materialized = build_cache(ptm)
    @test direct.left[1] == materialized.left[1]
    @test all(
        direct.left[i] ≈ materialized.left[i]
        for i in 2:length(direct.left)
    )
    @test direct.right[end] == materialized.right[end]
    @test all(
        direct.right[i] ≈ materialized.right[i]
        for i in 1:(length(direct.right) - 1)
    )
    @test rebuild_left!(materialized, bra, ket) === materialized
    @test rebuild_right!(materialized, bra, ket) === materialized
    @test rebuild!(materialized, bra, ket) === materialized
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
    @test transfer_at(mpsA, mpsB, 2) ≈ ptm[2]
    @test transfer_at(mpsB, mpsA, 2) ≈
          DenseTransferMatrix(mpsB[3], mpsB[4], mpsA[2])

    mpsB_short = random_mps(ComplexF64, 3, d, D)
    @test_throws DimensionMismatch PeriodicTransferMatrix(mpsA, mpsB_short)
end

@testset "PeriodicTransferMatrix — truncated construction and cache" begin
    D, d, N = 2, 2, 4
    trunc = truncrank(2)
    bra = random_mps(ComplexF64, N, d, D)
    ket = random_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(bra, ket, trunc)

    @test ptm isa PeriodicTransferMatrix{<:TruncatedTransferMatrix}
    @test all(transfer -> transfer.trunc === trunc, ptm)
    @test all(
        dense(ptm[i]) ≈
        dense(TruncatedTransferMatrix(bra[i], ket[i], trunc))
        for i in eachindex(ptm)
    )
    @test dense(transfer_at(ket, 2, trunc)) ≈
          dense(TruncatedTransferMatrix(ket[2], trunc))
    @test dense(transfer_at(ket, 2, TruncatedTransferPolicy(trunc))) ≈
          dense(TruncatedTransferMatrix(ket[2], trunc))
    @test dense(transfer_at(bra, ket, 2, trunc)) ≈ dense(ptm[2])
    @test expectation(bra, ket, trunc) ≈ expectation(ptm)

    identity_transfer = edge_identity(ptm)
    @test identity_transfer isa FormalIdentity{eltype(ptm)}
    @test identity_transfer.space == codomain(ptm[1])

    cache = build_cache(ptm)
    direct = build_cache(bra, ket, trunc)
    @test edge_space(cache) == edge_space(ptm)
    @test edge_space(cache) == codomain(ptm[1])
    @test all(transfer -> transfer isa eltype(ptm), cache.left[2:end])
    @test all(transfer -> transfer isa eltype(ptm), cache.right[1:end-1])
    @test dense(cache.left[3]) ≈ dense(ptm[1] * ptm[2])
    @test dense(cache.right[2]) ≈ dense(ptm[3] * ptm[4])
    @test all(
        dense(direct.left[i]) ≈ dense(cache.left[i])
        for i in 2:length(cache.left)
    )
    @test rebuild_left!(direct, bra, ket, trunc) === direct
    @test rebuild_right!(direct, bra, ket, trunc) === direct
    @test rebuild!(direct, bra, ket, trunc) === direct

    left_slot = cache.left[2]
    rank_one = TruncatedTransferMatrix(
        one_mps(ComplexF64, N, d, D)[1], truncrank(1)
    )
    replacement = TruncatedTransferMatrix(
        rank_one.u, rank_one.s, rank_one.vᴴ, trunc
    )
    advance!(cache, replacement, SweepState{Forward}(1, 0))
    @test cache.left[2] !== left_slot
    @test dense(cache.left[2]) ≈ dense(replacement)

    self_ptm = PeriodicTransferMatrix(ket, trunc)
    @test all(transfer -> transfer isa TruncatedTransferMatrix, self_ptm)
    @test expectation(ket, trunc) ≈ expectation(self_ptm)
end

@testset "PeriodicTransferMatrix — truncated mixed construction" begin
    D, d = 2, 2
    trunc = truncrank(2)
    mpsB = random_mps(ComplexF64, 4, d, D)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, d, D))

    ab = PeriodicTransferMatrix(mpsA, mpsB, trunc)
    ba = PeriodicTransferMatrix(mpsB, mpsA, trunc)

    @test all(transfer -> transfer isa TruncatedTransferMatrix, ab)
    @test all(transfer -> transfer isa TruncatedTransferMatrix, ba)
    @test dense(ab[1]) ≈
          dense(TruncatedTransferMatrix(mpsA[1], mpsB[1], mpsB[2], trunc))
    @test dense(ba[1]) ≈
          dense(TruncatedTransferMatrix(mpsB[1], mpsB[2], mpsA[1], trunc))
    @test dense(transfer_at(mpsA, mpsB, 2, trunc)) ≈ dense(ab[2])
    @test dense(transfer_at(mpsB, mpsA, 2, trunc)) ≈ dense(ba[2])

    mpsB_short = random_mps(ComplexF64, 3, d, D)
    @test_throws DimensionMismatch PeriodicTransferMatrix(mpsA, mpsB_short, trunc)
end

@testset "PeriodicTransferMatrix — truncated Z2 cache" begin
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mps = random_mps(ComplexF64, 3, P, V)
    ptm = PeriodicTransferMatrix(mps, truncrank(3))
    cache = build_cache(ptm)

    @test all(transfer -> transfer isa TruncatedTransferMatrix, ptm)
    @test edge_identity(ptm) ==
          FormalIdentity{eltype(ptm)}(codomain(ptm[1]))
    @test dense(cache.left[3]) ≈ dense(ptm[1] * ptm[2])
end

@testset "full-period product and expectation — 1-site" begin
    N, d, D = 3, 2, 2
    mps = random_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)

    E1, E2, E3 = ptm[1], ptm[2], ptm[3]
    @test *(ptm) ≈ E1 * E2 * E3
    @test expectation(ptm) ≈ overlap(mps, mps)
    @test expectation(mps) ≈ expectation(ptm)

    ket = random_mps(ComplexF64, N, d, D)
    @test expectation(PeriodicTransferMatrix(mps, ket)) ≈ overlap(mps, ket)
    @test expectation(mps, ket) ≈ overlap(mps, ket)
end

@testset "full-period product and expectation — one_mps analytic" begin
    N, d, D = 4, 2, 2
    mps = one_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)
    @test expectation(ptm) ≈ d^N * D^(2N)
    @test expectation(mps) ≈ expectation(ptm)
end

@testset "full-period product and expectation — 2-site MPS" begin
    N, d, D = 2, 2, 2
    mps = random_mps2(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)

    @test length(ptm) == N
    @test expectation(ptm) ≈ overlap(mps, mps)
    @test expectation(mps) ≈ expectation(ptm)
    @test *(ptm) ≈ ptm[1] * ptm[2]
end

@testset "expectation — mixed 2-site bra / 1-site ket" begin
    D, d = 2, 2
    mpsB = random_mps(ComplexF64, 4, d, D)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, d, D))
    ptm = PeriodicTransferMatrix(mpsA, mpsB)

    @test expectation(ptm) ≈ overlap(mpsA, mpsB)
    @test expectation(mpsA, mpsB) ≈ expectation(ptm)
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
