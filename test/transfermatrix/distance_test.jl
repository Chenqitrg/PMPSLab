@testset "dot — agrees with expectation of PeriodicTransferMatrix" begin
    N, d, D = 3, 2, 2
    bra = random_mps(ComplexF64, N, d, D)
    ket = random_mps(ComplexF64, N, d, D)

    @test dot(bra, ket) ≈ expectation(PeriodicTransferMatrix(bra, ket))
    @test dot(bra, ket) ≈ overlap(bra, ket)
    @test dot(bra, bra) ≈ expectation(PeriodicTransferMatrix(bra))
    @test imag(dot(bra, bra)) ≈ 0 atol=1e-12
    @test real(dot(bra, bra)) > 0
end

@testset "norm — real, non-negative, analytic for one_mps" begin
    N, d, D = 4, 2, 2
    mps = random_mps(ComplexF64, N, d, D)
    @test norm(mps)^2 ≈ dot(mps, mps)
    @test squared_norm(mps) ≈ real(dot(mps, mps))
    @test norm(mps) > 0
    @test imag(norm(mps)) ≈ 0 atol=1e-12

    mps1 = one_mps(ComplexF64, N, d, D)
    @test norm(mps1) ≈ sqrt(d^N * D^(2N))
end

@testset "negative squared quantities reject non-roundoff values" begin
    @test PMPSLab._nonnegative_squared(-eps(Float64), 1.0, "test quantity") == 0.0
    @test_throws DomainError PMPSLab._nonnegative_squared(-1.0, 1.0, "test quantity")
end

@testset "norm — 2-site MPS" begin
    N, d, D = 2, 2, 2
    mps = random_mps2(ComplexF64, N, d, D)
    @test norm(mps)^2 ≈ expectation(PeriodicTransferMatrix(mps))
    @test norm(mps)^2 ≈ real(overlap(mps, mps)) rtol=1e-10
end

@testset "diff_norm — zero for identical, positive for distinct" begin
    N, d, D = 3, 2, 2
    mps1 = random_mps(ComplexF64, N, d, D)
    mps2 = random_mps(ComplexF64, N, d, D)

    @test diff_norm(mps1, mps1) ≈ 0 atol=1e-12
    @test diff_norm(mps1, mps2) > 0
    @test diff_norm(mps1, mps2) ≈ diff_norm(mps2, mps1)
    @test squared_distance(mps1, mps2) ≈ diff_norm(mps1, mps2)
end

@testset "isapprox — gauge-equivalent states tolerate contraction roundoff" begin
    Random.seed!(310)
    mps = random_mps(ComplexF64, 4, 2, 3)
    gauged = PeriodicMPS(copy.(mps.tensors))
    gaugefix!(gauged, ProjectorCanonical(; tol=1e-12, maxiter=100))

    @test squared_distance(mps, gauged) >= 0
    @test isapprox(mps, gauged)

    different = deepcopy(mps)
    different[1] = 1.001 * different[1]
    @test !isapprox(mps, different)
end

@testset "diff_norm — with Z2-graded spaces" begin
    N = 3
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mps1 = random_mps(ComplexF64, N, P, V)
    mps2 = random_mps(ComplexF64, N, P, V)

    @test diff_norm(mps1, mps1) ≈ 0 atol=1e-12
    @test diff_norm(mps1, mps2) > 0
    @test diff_norm(mps1, mps2) ≈ diff_norm(mps2, mps1)

    d12 = norm(dense(mps1) - dense(mps2))
    @test d12^2 ≈ diff_norm(mps1, mps2) rtol=1e-10
end
