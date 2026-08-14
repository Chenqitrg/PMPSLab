# Directional-advance protocol tests. Shared fixtures come from circularupdate_test.jl.

@testset "advance! — validated directional refresh" begin
    # Forward update refreshes exactly one prefix slot.
    product = random_periodic_product(3, 5)
    new2 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    product2 = MatrixPeriodicProduct([i == 2 ? new2 : product[i] for i in 1:5])
    v = build_cache(product)
    left_slot = v.left[3]
    advance!(v, new2, SweepState{Forward}(2, 0))
    @test v.left[3] === left_slot
    @test environment(v, 3) ≈ naive_complement(product2, 3)

    # Reverse update refreshes exactly one suffix slot.
    new4 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    product4 = MatrixPeriodicProduct([i == 4 ? new4 : product[i] for i in 1:5])
    v = build_cache(product)
    right_slot = v.right[3]
    advance!(v, new4, SweepState{Backward}(4, 0))
    @test v.right[3] === right_slot
    @test environment(v, 3) ≈ naive_complement(product4, 3)

    # A periodic wrap explicitly invalidates and rebuilds the opposite direction.
    new5 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    product5 = MatrixPeriodicProduct([i == 5 ? new5 : product[i] for i in 1:5])
    v = build_cache(product)
    right_slots = copy(v.right)
    invalidate!(v, 5)
    @test_throws StaleCacheError environment(v, 1)
    rebuild_right!(v, product5)
    @test all(v.right[i] === right_slots[i] for i in eachindex(v.right))
    @test environment(v, 1) ≈ naive_complement(product5, 1)

    # advance! never hides the boundary rebuild cost.
    rejected = build_cache(product)
    @test_throws ArgumentError advance!(
        rejected, new5, SweepState{Forward}(5, 0)
    )
    @test_throws StaleCacheError environment(rejected, 1)

    # Forward site 1 -> site 2 (the prefix is an in-place copy).
    new1 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    product1 = MatrixPeriodicProduct([i == 1 ? new1 : product[i] for i in 1:5])
    v = build_cache(product)
    left_slot = v.left[2]
    advance!(v, new1, SweepState{Forward}(1, 0))
    @test v.left[2] === left_slot
    @test environment(v, 2) ≈ naive_complement(product1, 2)

    # Reverse periodic wrap follows the same complete-product rule.
    v = build_cache(product)
    left_slots = copy(v.left)
    invalidate!(v, 1)
    rebuild_left!(v, product1)
    @test all(v.left[i] === left_slots[i] for i in eachindex(v.left))
    @test environment(v, 5) ≈ naive_complement(product1, 5)

    # The factor type must match the product/cache factor type.
    @test_throws MethodError advance!(v, 1, SweepState{Forward}(1, 0))

    single = random_periodic_product(3, 1)
    single_cache = build_cache(single)
    @test advance!(single_cache, new1, SweepState{Forward}(1, 0)) === single_cache
end

@testset "cache validity — stale state is detected without contraction" begin
    N, D = 4, 3
    product = random_periodic_product(D, N)
    new3 = TensorMap(randn(ComplexF64, D, D), ℂ^D, ℂ^D)
    new2 = TensorMap(randn(ComplexF64, D, D), ℂ^D, ℂ^D)
    expected = MatrixPeriodicProduct([product[1], new2, new3, product[4]])
    v = build_cache(product)

    # This was the original stale-cache counterexample: forward updates in the
    # order 3 -> 2 used to leave left[3] silently readable as valid.
    advance!(v, new3, SweepState{Forward}(3, 0))
    advance!(v, new2, SweepState{Forward}(2, 0))

    @test v.left_valid_through == 2
    @test v.right_valid_from == 3
    @test_throws StaleCacheError environment(v, 4)
    @test_throws StaleCacheError v[1:3]

    # Environments whose required halves remain valid are still readable.
    @test environment(v, 3) ≈ naive_complement(expected, 3)

    # Explicit rebuild is the only operation here allowed to pay the full
    # contraction cost after an unsupported transition.
    @test rebuild!(v, expected) === v
    @test environment(v, 4) ≈ naive_complement(expected, 4)

    # Repeating the same directional site update remains an O(1) legal step.
    repeated = build_cache(product)
    advance!(repeated, new2, SweepState{Forward}(2, 0))
    newer2 = TensorMap(randn(ComplexF64, D, D), ℂ^D, ℂ^D)
    advance!(repeated, newer2, SweepState{Forward}(2, 0))
    expected_repeated = MatrixPeriodicProduct(
        [product[1], newer2, product[3], product[4]]
    )
    @test environment(repeated, 3) ≈ naive_complement(expected_repeated, 3)

    # Invalidation itself is O(1) and immediately protects affected reads.
    invalidate!(repeated, 4)
    @test repeated.right_valid_from == 4
    @test_throws StaleCacheError environment(repeated, 1)
    @test_throws StaleCacheError repeated[2:4]

    @test_throws BoundsError advance!(
        repeated, newer2, SweepState{Forward}(0, 0)
    )
end

@testset "directional rebuild — direct element generation" begin
    N, D = 4, 3
    product = random_periodic_product(D, N)
    cache = build_cache(product)

    new4 = TensorMap(randn(ComplexF64, D, D), ℂ^D, ℂ^D)
    after4 = MatrixPeriodicProduct([
        product[1], product[2], product[3], new4,
    ])
    right_calls = zeros(Int, N)
    @test rebuild_right!(
        cache,
        site -> begin
            right_calls[site] += 1
            copy(after4[site])
        end,
    ) === cache
    @test right_calls == [0, 1, 1, 1]
    @test environment(cache, 1) ≈ naive_complement(after4, 1)

    new1 = TensorMap(randn(ComplexF64, D, D), ℂ^D, ℂ^D)
    after1 = MatrixPeriodicProduct([
        new1, product[2], product[3], new4,
    ])
    left_calls = zeros(Int, N)
    @test rebuild_left!(
        cache,
        site -> begin
            left_calls[site] += 1
            copy(after1[site])
        end,
    ) === cache
    @test left_calls == [1, 1, 1, 0]
    @test environment(cache, N) ≈ naive_complement(after1, N)
end
