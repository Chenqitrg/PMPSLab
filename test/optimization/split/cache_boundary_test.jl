using PMPSLab: SplitEnvironmentPolicy
@testset "periodic boundary refresh includes the last changed tensor" begin
    for direction in (Forward, Backward)
        for policy in (DenseTransferPolicy(), TruncatedTransferPolicy(; trunc=TruncationStrategy()))
            approximation = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
            target = contract_pairs(approximation)
            policies = SplitEnvironmentPolicy(; normal=policy, mixed=policy)
            ctx = test_split_state(target, approximation; environments=policies)
            site = direction === Forward ? 4 : 1
            next_site = direction === Forward ? 1 : 4
            ctx.approximation[site] = 2 * ctx.approximation[site]
            PMPSLab._refresh_split_caches!(ctx, SweepState{direction}(site, 0))

            normal = build_transfer_cache(ctx.approximation, ctx.approximation, policy)
            mixed = build_transfer_cache(ctx.approximation, ctx.target, policy)
            @test dense(environment(ctx.normal_cache, next_site)) ≈
                  dense(environment(normal, next_site))
            mixed_site = target_site_for(next_site)
            @test dense(environment(ctx.mixed_cache, mixed_site)) ≈
                  dense(environment(mixed, mixed_site))
        end
    end
end
