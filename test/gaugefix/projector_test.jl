@testset "P_decomp — PR * PL ≈ I (ComplexF64)" begin
    D = 4
    L = TensorMap(randn(ComplexF64, D, D), ℂ^D, ℂ^D)
    R = TensorMap(randn(ComplexF64, D, D), ℂ^D, ℂ^D)
    PR, PL = P_decomp(R, L, MPSKit.TruncationStrategy())
    @plansor recon[-1; -2] := PR[-1; 1] * PL[1; -2]
    @test norm(recon - id(domain(R))) < 1e-10
end

@testset "P_decomp — PR * PL ≈ I (Float64)" begin
    D = 4
    L = TensorMap(randn(Float64, D, D), ℝ^D, ℝ^D)
    R = TensorMap(randn(Float64, D, D), ℝ^D, ℝ^D)
    PR, PL = P_decomp(R, L, MPSKit.TruncationStrategy())
    @plansor recon[-1; -2] := PR[-1; 1] * PL[1; -2]
    @test norm(recon - id(domain(R))) < 1e-10
end

@testset "P_decomp — PR * PL ≈ I (Z2Irrep)" begin
    V = Z2Space(0 => 2, 1 => 2)
    L = rand(ComplexF64, V ← V)
    R = rand(ComplexF64, V ← V)
    PR, PL = P_decomp(R, L, MPSKit.TruncationStrategy())
    @plansor recon[-1; -2] := PR[-1; 1] * PL[1; -2]
    @test norm(recon - id(domain(R))) < 1e-10
end

@testset "rank-deficient projectors compress the unsupported virtual subspace" begin
    mps = one_mps(ComplexF64, 4, 2, 3)
    dense_before = dense(mps)
    PR, PL = find_projectors(mps, ProjectorGaugeFix(; tol=1e-12, maxiter=100))

    for bond in 1:length(mps)
        pos = half(2bond + 1)
        support_projector = PR[pos] * PL[pos]
        @test support_projector * support_projector ≈ support_projector atol=1e-10
        @test !(support_projector ≈ id(domain(PR[pos])))
    end

    gaugefix!(mps, ProjectorGaugeFix(; tol=1e-12, maxiter=100))
    @test dense(mps) ≈ dense_before atol=1e-10
end

function structurally_rank_deficient_mps()
    return PeriodicMPS([
        rand(ComplexF64, ℂ^3 ⊗ ℂ^2 ← ℂ^1),
        rand(ComplexF64, ℂ^1 ⊗ ℂ^2 ← ℂ^3),
        rand(ComplexF64, ℂ^3 ⊗ ℂ^2 ← ℂ^1),
        rand(ComplexF64, ℂ^1 ⊗ ℂ^2 ← ℂ^3),
    ])
end

@testset "structural rank cleanup precedes projector gauge fixing" begin
    algorithm = ProjectorGaugeFix(; tol=1e-12, maxiter=100)

    # `find_projectors` is non-mutating and therefore cannot repair invalid
    # local bond dimensions. It should reject them before entering MPSKit's
    # fixed-workspace orthogonalization.
    direct = structurally_rank_deficient_mps()
    @test_throws ArgumentError find_projectors(direct, algorithm)

    # The mutating workflow may repair those dimensions exactly before finding
    # projectors. This is the preprocessing performed by InfiniteMPS as well.
    mps = structurally_rank_deficient_mps()
    dense_before = dense(mps)
    @test gaugefix!(mps, algorithm) === mps
    @test all(MPSKit.isfullrank, mps)
    @test [dim(right_virtualspace(tensor)) for tensor in mps] == [1, 2, 1, 2]
    @test norm(dense(mps) - dense_before) / norm(dense_before) < 1e-12
end
