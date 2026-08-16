function test_factored_split_environments(mpsA, mpsB)
    ctx = SplitContext(mpsA, mpsB)

    for site in (1, 2)
        pa = target_site_for(site)
        x = transpose(current_approximation(ctx, site), ((2,), (1, 3)))
        A = target_tensor(ctx, site)
        paired_approximationᴴ = paired_approximation(ctx, site)'

        dense_BB = environment(ctx.normal_cache, site)
        truncated_BB = TruncatedTransferMatrix(
            dense_BB, TruncationStrategy()
        )
        expected_action = PMPSLab._split_norm_action(x, dense_BB)
        @test PMPSLab._split_norm_action(x, truncated_BB) ≈
              expected_action rtol=1e-12

        dense_AB = environment(ctx.mixed_cache, pa)
        truncated_AB = TruncatedTransferMatrix(
            dense_AB, TruncationStrategy()
        )
        expected_rhs = PMPSLab._split_rhs(dense_AB, A, paired_approximationᴴ, site)
        @test PMPSLab._split_rhs(truncated_AB, A, paired_approximationᴴ, site) ≈
              expected_rhs rtol=1e-12
    end
    return nothing
end

@testset "split environments — truncated factored contractions" begin
    mpsB = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    mpsA = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, 2, 2))
    test_factored_split_environments(mpsA, mpsB)
end

@testset "split environments — truncated Z2 contractions" begin
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsB = rand(PeriodicMPS, ComplexF64, 4, P, V)
    mpsA = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, P, V))
    test_factored_split_environments(mpsA, mpsB)
end

@testset "split environments — truncated Krylov solver" begin
    mpsB = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    mpsA = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, 2, 2))
    ctx = SplitContext(mpsA, mpsB)
    site = 1
    pa = target_site_for(site)

    dense_BB = environment(ctx.normal_cache, site)
    truncated_BB = TruncatedTransferMatrix(dense_BB, TruncationStrategy())
    dense_AB = environment(ctx.mixed_cache, pa)
    truncated_AB = TruncatedTransferMatrix(dense_AB, TruncationStrategy())
    x0 = transpose(current_approximation(ctx, site), ((2,), (1, 3)))
    W = PMPSLab._split_rhs(
        truncated_AB, target_tensor(ctx, site), paired_approximation(ctx, site)', site
    )
    solver = KrylovSolve(; algorithm=GMRES(; maxiter=100, tol=1e-12))

    expected = PMPSLab._solve_split_norm(solver, x0, dense_BB, W)
    actual = PMPSLab._solve_split_norm(solver, x0, truncated_BB, W)
    @test actual ≈ expected rtol=1e-10

    direct_expected = PMPSLab._solve_split_norm(
        CholeskySolve(), x0, dense_BB, W
    )
    direct_actual = PMPSLab._solve_split_norm(
        CholeskySolve(), x0, truncated_BB, W
    )
    @test direct_actual ≈ direct_expected rtol=1e-10
end
