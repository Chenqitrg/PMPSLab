# Tests for src/circularupdate/cache.jl — the RingCache partial-product cache
# for periodic LoopMonoid products.
#
# Reference: products are verified against naive left-to-right ordered
# multiplications. Square random matrices make `*` noncommutative, so any
# ordering mistake in the cache shows up as a failing test.

using Test
using PeriodicMPS:
    PeriodicMPS,
    random_mps,
    LoopMonoid,
    LoopTransferMatrix,
    RingCache,
    build_cache,
    advance!,
    environment,
    left_cache,
    left_right_space,
    left_right_unit,
    range_prod,
    right_cache,
    ZipSweep,
    CircularSweep
using TensorKit
using MPSKit: PeriodicVector

# Minimal LoopMonoid subtype for testing: a plain vector of square matrices.
struct MatrixLoop{T} <: LoopMonoid
    sites::Vector{T}
end

Base.length(v::MatrixLoop) = length(v.sites)
Base.getindex(v::MatrixLoop, i::Int) = v.sites[i]

# Random loop of D×D matrices with trivial (ℂ^D) bond spaces.
random_loop(D::Int, N::Int) = MatrixLoop(
    [TensorMap(randn(ComplexF64, D, D), ℂ^D, ℂ^D) for _ in 1:N]
)

# Random loop of symmetric matrices with bond space V.
random_loop(V::ElementarySpace, N::Int) = MatrixLoop(
    [rand(ComplexF64, V ← V) for _ in 1:N]
)

# Ordered product of a (non-wrapping) subrange, computed directly.
function naive_range_prod(loop, r)
    acc = loop[r[1]]
    for i in (first(r) + 1):last(r)
        acc = acc * loop[i]
    end
    return acc
end

# Product of everything except `site`, in the order
#   w[site+1] * ... * w[N] * w[1] * ... * w[site-1]
# (the same order in which `environment` composes right half then left half).
function naive_complement(loop, site)
    N = length(loop)
    acc = loop[mod1(site + 1, N)]
    for offset in 2:(N - 1)
        acc = acc * loop[mod1(site + offset, N)]
    end
    return acc
end

@testset "build_cache — cache contents vs naive products" begin
    for N in [2, 3, 5, 8], site in 1:N
        loop = random_loop(3, N)
        v = build_cache(loop, site)

        @test v isa RingCache
        @test length(v) == N
        @test lastindex(v) == N
        @test scalartype(v) == ComplexF64

        # Prefix ranges v[1:k] with k < site.
        for k in 1:(site - 1)
            @test v[1:k] ≈ naive_range_prod(loop, 1:k)
        end

        # Suffix ranges v[k:N] with k > site.
        for k in (site + 1):N
            @test v[k:N] ≈ naive_range_prod(loop, k:N)
        end
    end
end

@testset "range queries — invalid ranges throw" begin
    loop = random_loop(3, 5)
    v = build_cache(loop, 3)

    # Any range that crosses the cut site is not representable.
    @test_throws ArgumentError v[1:3]   # contains the cut
    @test_throws ArgumentError v[2:3]   # contains the cut
    @test_throws ArgumentError v[2:4]   # crosses the cut
    @test_throws ArgumentError v[3:3]   # the cut site itself
    @test_throws ArgumentError v[3:4]   # starts at the cut
    @test_throws ArgumentError v[4:4]   # suffix but does not end at N
    @test_throws ArgumentError v[2:1]   # empty range

    # Valid: pure prefix (ends before the cut) and pure suffix (starts after).
    @test v[1:2] ≈ naive_range_prod(loop, 1:2)
    @test v[4:5] ≈ naive_range_prod(loop, 4:5)
end

@testset "left/right cache and environment — every cut position" begin
    N, D = 5, 3
    loop = random_loop(D, N)

    for site in 1:N
        v = build_cache(loop, site)

        expected_left = site == 1 ?
                        left_right_unit(v) :
                        naive_range_prod(loop, 1:(site - 1))
        expected_right = site == N ?
                         left_right_unit(v) :
                         naive_range_prod(loop, (site + 1):N)

        @test left_cache(v) ≈ expected_left
        @test right_cache(v) ≈ expected_right

        # The environment is the product of everything except the cut site,
        # in right-then-left order (noncommutative: order matters).
        @test environment(v) ≈ naive_complement(loop, site)
        @test environment(v) ≈ right_cache(v) * left_cache(v)
    end

    # The default site argument is 1.
    v_default = build_cache(loop)
    @test environment(v_default) ≈ naive_complement(loop, 1)
end

@testset "units and spaces" begin
    D = 3
    loop = random_loop(D, 4)

    # site > 1: the unit lives at the codomain of the first prefix product.
    # (TensorKit wraps single legs in a ProductSpace, hence the comparison.)
    @test left_right_space(build_cache(loop, 2)) == ProductSpace(ℂ^D)
    # site == 1: the unit lives at the domain of the last suffix product.
    @test left_right_space(build_cache(loop, 1)) == ProductSpace(ℂ^D)
    @test dim(left_right_space(build_cache(loop, 2))) == D
    @test left_right_unit(build_cache(loop, 2)) ≈ id(ComplexF64, ℂ^D)

    # Empty halves return the category unit.
    @test left_cache(build_cache(loop, 1)) ≈ id(ComplexF64, ℂ^D)
    @test right_cache(build_cache(loop, 4)) ≈ id(ComplexF64, ℂ^D)
end

@testset "Z2Irrep — unit space and environment" begin
    V = Z2Space(0 => 2, 1 => 2)
    loop = random_loop(V, 4)

    v = build_cache(loop, 2)
    @test left_right_space(v) == ProductSpace(V)
    @test left_right_unit(v) ≈ id(ComplexF64, V)

    v1 = build_cache(loop, 1)
    @test left_right_space(v1) == ProductSpace(V)

    for site in 1:4
        @test environment(build_cache(loop, site)) ≈ naive_complement(loop, site)
    end
end

@testset "Float64 element type" begin
    loop = MatrixLoop([TensorMap(randn(Float64, 2, 2), ℝ^2, ℝ^2) for _ in 1:4])
    v = build_cache(loop, 2)
    @test scalartype(v) == Float64
    @test v[1:1] ≈ loop[1]
    @test v[3:4] ≈ naive_range_prod(loop, 3:4)
    @test environment(v) ≈ naive_complement(loop, 2)
end

@testset "LoopTransferMatrix integration" begin
    N, d, D = 4, 2, 2
    mps = random_mps(ComplexF64, N, d, D)
    ltm = LoopTransferMatrix(mps)
    v = build_cache(ltm, 2)

    @test length(v) == N
    # Transfer matrices live in (L ⊗ L̄) ← (R ⊗ R̄), with L = R = ℂ^D here
    # (L̄ is the dual space).
    @test left_right_space(v) == ℂ^D ⊗ (ℂ^D)'

    # Every representable range matches a direct ordered product.
    @test v[1:1] ≈ range_prod(ltm, 1:1)
    @test v[3:4] ≈ range_prod(ltm, 3:4)
    @test v[4:4] ≈ range_prod(ltm, 4:4)

    @test left_cache(v) ≈ range_prod(ltm, 1:1)
    @test right_cache(v) ≈ range_prod(ltm, 3:4)
    @test environment(v) ≈ naive_complement(ltm, 2)

    # Sanity: the full loop product helper agrees with a direct product.
    @test *(ltm) ≈ naive_range_prod(ltm, 1:N)
end

@testset "ZipSweep — 往返顺序" begin
    # 1, 2, …, N, N-1, …, 2（不重复访问 1）
    @test collect(site for (site, _) in ZipSweep(4)) == [1, 2, 3, 4, 3, 2]
    @test collect(site for (site, _) in ZipSweep(5)) == [1, 2, 3, 4, 5, 4, 3, 2]
    # 掉头处 dir 变号：前滑 +1，掉头后 -1
    @test collect(dir for (_, dir) in ZipSweep(4)) == [1, 1, 1, -1, -1, -1]
    # N = 1 只访问一次
    @test collect(site for (site, _) in ZipSweep(1)) == [1]
end

@testset "CircularSweep — 循环与最大趟数" begin
    cs = CircularSweep(4; maximal_sweep = 2)
    @test collect(site for (site, _) in cs) == [1, 2, 3, 4, 1, 2, 3, 4]
    @test cs.n_sweep == 2
    # 单向循环：dir 恒为 +1
    @test all(dir == 1 for (_, dir) in CircularSweep(4; maximal_sweep = 2))
    # 一趟 N 个 site
    @test length(collect(CircularSweep(3; maximal_sweep = 1))) == 3
    @test length(collect(CircularSweep(1; maximal_sweep = 3))) == 3
end

@testset "advance! — 滑动缓存" begin
    # 前滑：cut 2 更新 site 2 → cut 3
    loop = random_loop(3, 5)
    new2 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    loop2 = MatrixLoop([i == 2 ? new2 : loop[i] for i in 1:5])
    v = build_cache(loop, 2)
    advance!(v, loop2, new2, 1)
    @test v.site == 3
    @test environment(v) ≈ naive_complement(loop2, 3)

    # 后滑：cut 4 更新 site 4 → cut 3
    new4 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    loop4 = MatrixLoop([i == 4 ? new4 : loop[i] for i in 1:5])
    v = build_cache(loop, 4)
    advance!(v, loop4, new4, -1)
    @test v.site == 3
    @test environment(v) ≈ naive_complement(loop4, 3)

    # 回卷 N→1：更新 site 5 后整个缓存重建
    new5 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    loop5 = MatrixLoop([i == 5 ? new5 : loop[i] for i in 1:5])
    v = build_cache(loop, 5)
    advance!(v, loop5, new5, 1)
    @test v.site == 1
    @test environment(v) ≈ naive_complement(loop5, 1)

    # 前滑 cut 1 → cut 2（无前缀可乘，直接写入）
    new1 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    loop1 = MatrixLoop([i == 1 ? new1 : loop[i] for i in 1:5])
    v = build_cache(loop, 1)
    advance!(v, loop1, new1, 1)
    @test v.site == 2
    @test environment(v) ≈ naive_complement(loop1, 2)
end
