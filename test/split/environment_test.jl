function test_factored_split_environments(mpsA, mpsB)
    ctx = SplitContext(mpsA, mpsB)

    for site in (1, 2)
        pa = pa_from_pb(site)
        x = transpose(this_B(ctx, site), ((2,), (1, 3)))
        A = this_A(ctx, site)
        other_Bᴴ = other_B(ctx, site)'

        dense_BB = environment(ctx.cacheBB, site)
        truncated_BB = TruncatedTransferMatrix(
            dense_BB, TruncationStrategy()
        )
        expected_action = PMPSLab._split_norm_action(x, dense_BB)
        @test PMPSLab._split_norm_action(x, truncated_BB) ≈
              expected_action rtol=1e-12

        dense_AB = environment(ctx.cacheAB, pa)
        truncated_AB = TruncatedTransferMatrix(
            dense_AB, TruncationStrategy()
        )
        expected_rhs = PMPSLab._split_rhs(dense_AB, A, other_Bᴴ, site)
        @test PMPSLab._split_rhs(truncated_AB, A, other_Bᴴ, site) ≈
              expected_rhs rtol=1e-12
    end
    return nothing
end

@testset "split environments — truncated factored contractions" begin
    mpsB = random_mps(ComplexF64, 4, 2, 2)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, 2, 2))
    test_factored_split_environments(mpsA, mpsB)
end

@testset "split environments — truncated Z2 contractions" begin
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsB = random_mps(ComplexF64, 4, P, V)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, P, V))
    test_factored_split_environments(mpsA, mpsB)
end

@testset "split environments — truncated Krylov solve" begin
    mpsB = random_mps(ComplexF64, 4, 2, 2)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, 2, 2))
    ctx = SplitContext(mpsA, mpsB)
    site = 1
    pa = pa_from_pb(site)

    dense_BB = environment(ctx.cacheBB, site)
    truncated_BB = TruncatedTransferMatrix(dense_BB, TruncationStrategy())
    dense_AB = environment(ctx.cacheAB, pa)
    truncated_AB = TruncatedTransferMatrix(dense_AB, TruncationStrategy())
    x0 = transpose(this_B(ctx, site), ((2,), (1, 3)))
    W = PMPSLab._split_rhs(
        truncated_AB, this_A(ctx, site), other_B(ctx, site)', site
    )
    solve = KrylovSolve(GMRES(; maxiter=100, tol=1e-12))

    expected = PMPSLab._solve_split_norm(solve, x0, dense_BB, W)
    actual = PMPSLab._solve_split_norm(solve, x0, truncated_BB, W)
    @test actual ≈ expected rtol=1e-10

    direct_expected = PMPSLab._solve_split_norm(
        CholeskySolve(), x0, dense_BB, W
    )
    direct_actual = PMPSLab._solve_split_norm(
        CholeskySolve(), x0, truncated_BB, W
    )
    @test direct_actual ≈ direct_expected rtol=1e-10
end
