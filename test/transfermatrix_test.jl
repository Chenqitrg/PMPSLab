# Tests for src/transfermatrix.jl — dense transfer matrices over one period.

using Test
using Random
using PMPSLab:
    PeriodicMPS,
    random_mps,
    one_mps,
    random_mps2,
    pair_contraction,
    dense,
    site_transfer_matrix,
    PeriodicTransferMatrix,
    edge_identity,
    expectation,
    squared_norm,
    squared_distance,
    diff_norm,
    gaugefix!,
    ProjectorCanonical
using TensorKit
using MPSKit: PeriodicVector, left_virtualspace

# Manual pairwise overlap ⟨bra|ket⟩ via the dense states: for one-site states
# the physical leg orders of dense(bra) and dense(ket) match positionally.
function overlap(bra::PeriodicMPS, ket::PeriodicMPS)
    # TensorKit.scalar extracts the value of a trivial-space (0-leg) map
    return scalar(dense(bra)' * dense(ket))
end

@testset "site_transfer_matrix — 1-site ket" begin
    D, d = 3, 2
    T = randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)
    E = site_transfer_matrix(T)

    # E: (L ⊗ L* ← R ⊗ R*), i.e. 2 in / 2 out legs, all bond spaces
    @test length(codomain(E)) == 2
    @test length(domain(E)) == 2
    @test codomain(E) == ℂ^D ⊗ (ℂ^D)'

    # manual: E[l l'; r r'] = Σ_p conj(T[l' p r']) · T[l p r]
    @plansor manual[-1 -2; -3 -4] := T[-1 1; -3] * T'[-4; -2 1]
    @test E ≈ manual
end

@testset "site_transfer_matrix — 2-site ket" begin
    D, d = 3, 2
    T = randn(ComplexF64, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D)
    E = site_transfer_matrix(T)

    @test length(codomain(E)) == 2
    @test length(domain(E)) == 2
    @test codomain(E) == ℂ^D ⊗ (ℂ^D)'

    @plansor manual[-1 -2; -3 -4] := T[-1 1 2; -3] * T'[-4; -2 1 2]
    @test E ≈ manual
end

@testset "site_transfer_matrix — mixed (2-site bra, two 1-site kets)" begin
    D, d = 3, 2
    A = randn(ComplexF64, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D)     # bra: one 2-site tensor
    B1 = randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)          # ket: left half
    B2 = randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)          # ket: right half
    E = site_transfer_matrix(A, B1, B2)

    @test length(codomain(E)) == 2
    @test length(domain(E)) == 2
    @test codomain(E) == ℂ^D ⊗ (ℂ^D)'

    # manual: E[l l'; r r'] = Σ_{p,q,m} B1[l p; m] B2[m q; r] conj(A[l' q p'; r'])
    @plansor manual[-1 -2; -3 -4] := B1[-1 1; 2] * B2[2 3; -3] * A'[-4; -2 1 3]
    @test E ≈ manual
end

# ==============================================================================
# PeriodicTransferMatrix — construction and full-period products
# ==============================================================================

@testset "PeriodicTransferMatrix — construction from one MPS" begin
    N, d, D = 4, 2, 2
    mps = random_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)

    @test length(ptm) == N
    @test parent(ptm) === ptm.matrices
    @test eachindex(ptm) == eachindex(ptm.matrices)
    @test collect(ptm) == collect(ptm.matrices)
    for i in 1:N
        @test ptm[i] ≈ site_transfer_matrix(mps[i])
    end
    # getindex by HalfInt-style positions via Int; setindex! replaces a site
    new = site_transfer_matrix(random_mps(ComplexF64, 1, d, D)[1])
    ptm[2] = new
    @test ptm[2] ≈ new
end

@testset "PeriodicTransferMatrix — construction from (bra, ket)" begin
    N, d, D = 3, 2, 3
    bra = random_mps(ComplexF64, N, d, D)
    ket = random_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(bra, ket)

    @test length(ptm) == N
    @test ptm[2] ≈ site_transfer_matrix(bra[2], ket[2])

    # edge unit: identity on the bond space L ⊗ L*
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
    mpsB = random_mps(ComplexF64, 4, d, D)           # 1-site: 4 sites
    mpsA = pair_contraction(random_mps(ComplexF64, 4, d, D))  # 2-site: 2 sites
    ptm = PeriodicTransferMatrix(mpsA, mpsB)

    @test length(ptm) == 2
    @test ptm[1] ≈ site_transfer_matrix(mpsA[1], mpsB[1], mpsB[2])
    @test ptm[2] ≈ site_transfer_matrix(mpsA[2], mpsB[3], mpsB[4])

    # mismatched site counts must be rejected, not silently wrapped
    mpsB_short = random_mps(ComplexF64, 3, d, D)
    @test_throws DimensionMismatch PeriodicTransferMatrix(mpsA, mpsB_short)
end

@testset "full-period product and expectation — 1-site" begin
    N, d, D = 3, 2, 2
    mps = random_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)

    # *(ptm) equals the ordered product of the site transfer matrices
    E1, E2, E3 = ptm[1], ptm[2], ptm[3]
    @test *(ptm) ≈ E1 * E2 * E3

    # expectation = trace of the periodic product = ⟨mps|mps⟩
    @test expectation(ptm) ≈ overlap(mps, mps)

    # bra ≠ ket: the overlap is genuinely complex
    ket = random_mps(ComplexF64, N, d, D)
    @test expectation(PeriodicTransferMatrix(mps, ket)) ≈ overlap(mps, ket)
end

@testset "full-period product and expectation — one_mps analytic" begin
    N, d, D = 4, 2, 2
    mps = one_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)
    # all-ones tensors: each E = d·ones(D², D²), so
    # tr(∏E) = d^N · tr(ones^N) = d^N · D^{2N}
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

    # ⟨A|B⟩: A's two physical legs pair positionally with B's consecutive sites
    @test expectation(ptm) ≈ overlap(mpsA, mpsB)
end

@testset "expectation — Z2-graded spaces" begin
    N = 3
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    bra = random_mps(ComplexF64, N, P, V)
    ket = random_mps(ComplexF64, N, P, V)

    @test expectation(PeriodicTransferMatrix(bra, ket)) ≈ overlap(bra, ket)

    # graded one-site overlap with itself
    @test expectation(PeriodicTransferMatrix(bra)) ≈ overlap(bra, bra)
end

# ==============================================================================
# dot, norm, diff_norm
# ==============================================================================

@testset "dot — agrees with expectation of PeriodicTransferMatrix" begin
    N, d, D = 3, 2, 2
    bra = random_mps(ComplexF64, N, d, D)
    ket = random_mps(ComplexF64, N, d, D)

    @test dot(bra, ket) ≈ expectation(PeriodicTransferMatrix(bra, ket))
    @test dot(bra, ket) ≈ overlap(bra, ket)

    # same MPS: inner product = expectation (real, positive)
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

    # analytic: one_mps → each E = d·ones(D², D²), tr(∏E) = d^N · D^{2N}
    mps1 = one_mps(ComplexF64, N, d, D)
    @test norm(mps1) ≈ sqrt(d^N * D^(2N))
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
    # This seed produces a small positive cancellation error in
    # n₁ + n₂ - 2Re⟨ψ₁|ψ₂⟩, rather than the negative error that is
    # already clamped by squared_distance.
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

    # consistent with manual computation via dense
    d12 = norm(dense(mps1) - dense(mps2))
    @test d12^2 ≈ diff_norm(mps1, mps2) rtol=1e-10
end
