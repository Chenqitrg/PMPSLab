@testset "cache inputs are transfer factors or getters" begin
    mps = one_mps(ComplexF64, 4, 2, 2)
    @test_throws MethodError build_cache(mps)
    @test_throws MethodError build_cache(mps, mps)
    @test_throws MethodError build_cache(mps, mps, DenseTransferPolicy())
    for operation in (build_cache, rebuild!, rebuild_left!, rebuild_right!)
        @test all(method.module !== PMPSLab for method in methods(operation))
    end

    for policy in (DenseTransferPolicy(), TruncatedTransferPolicy(; trunc=TruncationStrategy()))
        state = copy(mps)
        requested = Int[]
        factor_at = site -> begin
            push!(requested, site)
            transfer_at(state, site, policy)
        end
        cache = build_cache(length(state), factor_at)
        factors = [transfer_at(state, site, policy) for site in eachindex(state)]
        reference = build_cache(factors)
        @test TensorMap(environment(cache, 2)) ≈ TensorMap(environment(reference, 2))
        @test_throws MethodError rebuild!(cache, state, state)

        empty!(requested)
        @test rebuild!(cache, factor_at) === cache
        @test isempty(requested)

        state[2] = 2 * state[2]
        invalidate!(cache, 2)
        @test_throws StaleCacheError environment(cache, 1)
        @test rebuild!(cache, factor_at) === cache
        @test requested == [2, 3, 2]
        updated = [transfer_at(state, site, policy) for site in eachindex(state)]
        @test TensorMap(environment(cache, 1)) ≈ TensorMap(updated[2] * updated[3] * updated[4])
    end
end

@testset "named PMPS transfer-cache adapters" begin
    mps = one_mps(ComplexF64, 4, 2, 2)
    shorter = one_mps(ComplexF64, 3, 2, 2)
    target = contract_pairs(mps)
    for operation in (build_transfer_cache, rebuild_transfer_cache!,
                      rebuild_left_transfer_cache!, rebuild_right_transfer_cache!)
        @test parentmodule(operation) === PMPSLab
        @test Base.ispublic(PMPSLab, nameof(operation))
    end
    @test environment(build_transfer_cache(mps), 2) ≈
          environment(build_cache(PeriodicTransferMatrix(mps)), 2)
    @test TensorMap(environment(build_transfer_cache(mps, TruncationStrategy()), 2)) ≈
          TensorMap(environment(build_cache(PeriodicTransferMatrix(mps)), 2))

    for policy in (DenseTransferPolicy(), TruncatedTransferPolicy(; trunc=TruncationStrategy()))
        state = copy(mps)
        cache = build_transfer_cache(state, state, policy)
        before = copy(environment(cache, 1))
        @test length(build_transfer_cache(target, state, policy)) == length(target)
        @test length(build_transfer_cache(state, target, policy)) == length(target)
        @test_throws DimensionMismatch build_transfer_cache(state, shorter, policy)
        @test_throws DimensionMismatch build_transfer_cache(target, shorter, policy)
        @test_throws DimensionMismatch rebuild_transfer_cache!(cache, shorter, shorter, policy)

        state[2] = 2 * state[2]
        invalidate!(cache, 2)
        @test rebuild_left_transfer_cache!(cache, state, state, policy) === cache
        @test_throws StaleCacheError environment(cache, 1)
        @test rebuild_right_transfer_cache!(cache, state, state, policy) === cache
        @test TensorMap(environment(cache, 1)) ≈ 4 * TensorMap(before)

        state[3] = 3 * state[3]
        invalidate!(cache, 3)
        @test rebuild_transfer_cache!(cache, state, state, policy) === cache
        @test TensorMap(environment(cache, 1)) ≈ 36 * TensorMap(before)
    end
end
