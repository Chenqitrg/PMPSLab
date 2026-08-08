# Tests for the PeriodicCache of AbstractPeriodicProduct values.
#
# Reference: products are verified against naive left-to-right ordered
# multiplications. Square random matrices make `*` noncommutative, so any
# ordering mistake in the cache shows up as a failing test.

using Test
using PMPSLab:
    PeriodicMPS,
    random_mps,
    AbstractPeriodicProduct,
    PeriodicTransferMatrix,
    PeriodicCache,
    build_cache,
    advance!,
    environment,
    left_cache,
    range_product,
    right_cache,
    edge_identity,
    SweepState,
    ZigzagSweep,
    CircularSweep
using TensorKit
using MPSKit: PeriodicVector

# Minimal periodic-product subtype for testing: a vector of square matrices.
struct MatrixPeriodicProduct{T} <: AbstractPeriodicProduct
    sites::Vector{T}
end

Base.length(v::MatrixPeriodicProduct) = length(v.sites)
Base.getindex(v::MatrixPeriodicProduct, i::Int) = v.sites[i]

# Identity on the common bond space.
PMPSLab.edge_identity(product::MatrixPeriodicProduct) =
    id(scalartype(product[1]), codomain(product[1]))

# Random periodic product of D×D matrices with trivial bond spaces.
random_periodic_product(D::Int, N::Int) = MatrixPeriodicProduct(
    [TensorMap(randn(ComplexF64, D, D), ℂ^D, ℂ^D) for _ in 1:N]
)

# Random periodic product with bond space V.
random_periodic_product(V::ElementarySpace, N::Int) = MatrixPeriodicProduct(
    [rand(ComplexF64, V ← V) for _ in 1:N]
)

# Ordered product of a (non-wrapping) subrange, computed directly.
function naive_range_product(product, r)
    acc = product[r[1]]
    for i in (first(r) + 1):last(r)
        acc = acc * product[i]
    end
    return acc
end

# Product of everything except `site`, in the order
#   w[site+1] * ... * w[N] * w[1] * ... * w[site-1]
# (the same order in which `environment` composes right half then left half).
function naive_complement(product, site)
    N = length(product)
    acc = product[mod1(site + 1, N)]
    for offset in 2:(N - 1)
        acc = acc * product[mod1(site + offset, N)]
    end
    return acc
end

@testset "build_cache — cache contents vs naive products" begin
    for N in [2, 3, 5, 8], site in 1:N
        product = random_periodic_product(3, N)
        v = build_cache(product, site)

        @test v isa PeriodicCache
        @test length(v) == N
        @test lastindex(v) == N
        @test scalartype(v.cache[1]) == ComplexF64

        # Prefix ranges v[1:k] with k < site.
        for k in 1:(site - 1)
            @test v[1:k] ≈ naive_range_product(product, 1:k)
        end

        # Suffix ranges v[k:N] with k > site.
        for k in (site + 1):N
            @test v[k:N] ≈ naive_range_product(product, k:N)
        end
    end
end

@testset "build_cache — validates the cut" begin
    product = random_periodic_product(3, 3)
    @test_throws BoundsError build_cache(product, 0)
    @test_throws BoundsError build_cache(product, 4)
    @test_throws ArgumentError build_cache(MatrixPeriodicProduct(TensorMap[]))
end

@testset "range queries — invalid ranges throw" begin
    product = random_periodic_product(3, 5)
    v = build_cache(product, 3)

    # Any range that crosses the cut site is not representable.
    @test_throws ArgumentError v[1:3]   # contains the cut
    @test_throws ArgumentError v[2:3]   # contains the cut
    @test_throws ArgumentError v[2:4]   # crosses the cut
    @test_throws ArgumentError v[3:3]   # the cut site itself
    @test_throws ArgumentError v[3:4]   # starts at the cut
    @test_throws ArgumentError v[4:4]   # suffix but does not end at N
    @test_throws ArgumentError v[2:1]   # empty range

    # Valid: pure prefix (ends before the cut) and pure suffix (starts after).
    @test v[1:2] ≈ naive_range_product(product, 1:2)
    @test v[4:5] ≈ naive_range_product(product, 4:5)
end

@testset "left/right cache and environment — every cut position" begin
    N, D = 5, 3
    product = random_periodic_product(D, N)

    for site in 1:N
        v = build_cache(product, site)

        expected_left = site == 1 ?
                        v.edge_identity :
                        naive_range_product(product, 1:(site - 1))
        expected_right = site == N ?
                         v.edge_identity :
                         naive_range_product(product, (site + 1):N)

        @test left_cache(v) ≈ expected_left
        @test right_cache(v) ≈ expected_right

        # The environment is the product of everything except the cut site,
        # in right-then-left order (noncommutative: order matters).
        @test environment(v) ≈ naive_complement(product, site)
        @test environment(v) ≈ right_cache(v) * left_cache(v)
    end

    # The default site argument is 1.
    v_default = build_cache(product, 1)
    @test environment(v_default) ≈ naive_complement(product, 1)
end

@testset "units and spaces" begin
    D = 3
    product = random_periodic_product(D, 4)
    v2 = build_cache(product, 2)
    @test v2.edge_identity ≈ id(ComplexF64, ℂ^D)
    @test dim(codomain(v2.edge_identity)) == D

    # Empty halves return the unit.
    @test left_cache(build_cache(product, 1)) ≈ edge_identity(product)
    @test right_cache(build_cache(product, 4)) ≈ edge_identity(product)
end

@testset "Z2Irrep — unit space and environment" begin
    V = Z2Space(0 => 2, 1 => 2)
    product = random_periodic_product(V, 4)
    v = build_cache(product, 2)
    @test v.edge_identity ≈ id(ComplexF64, V)

    for site in 1:4
        @test environment(build_cache(product, site)) ≈ naive_complement(product, site)
    end
end

@testset "Float64 element type" begin
    product = MatrixPeriodicProduct([TensorMap(randn(Float64, 2, 2), ℝ^2, ℝ^2) for _ in 1:4])
    v = build_cache(product, 2)
    @test v[1:1] ≈ product[1]
    @test v[3:4] ≈ naive_range_product(product, 3:4)
    @test environment(v) ≈ naive_complement(product, 2)
end

@testset "PeriodicTransferMatrix integration" begin
    N, d, D = 4, 2, 2
    mps = random_mps(ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)
    v = build_cache(ptm, 2)

    @test length(v) == N
    @test v.edge_identity ≈ id(ComplexF64, ℂ^D ⊗ (ℂ^D)')

    # Every representable range matches a direct ordered product.
    @test v[1:1] ≈ range_product(ptm, 1:1)
    @test v[3:4] ≈ range_product(ptm, 3:4)
    @test v[4:4] ≈ range_product(ptm, 4:4)

    @test left_cache(v) ≈ range_product(ptm, 1:1)
    @test right_cache(v) ≈ range_product(ptm, 3:4)
    @test environment(v) ≈ naive_complement(ptm, 2)

    # The full-period helper agrees with a direct product.
    @test *(ptm) ≈ naive_range_product(ptm, 1:N)
end

@testset "ZigzagSweep — 往返顺序" begin
    # 1, 2, …, N, N-1, …, 2（不重复访问 1）
    @test collect(st.site for st in ZigzagSweep(4; maximal_sweep=1)) == [1, 2, 3, 4, 3, 2]
    @test collect(st.site for st in ZigzagSweep(5; maximal_sweep=1)) == [1, 2, 3, 4, 5, 4, 3, 2]
    # 掉头处 dir 变号：前滑 +1，掉头后 -1
    @test collect(st.dir for st in ZigzagSweep(4; maximal_sweep=1)) == [1, 1, 1, -1, -1, -1]
    # N = 1 只访问一次
    @test collect(st.site for st in ZigzagSweep(1; maximal_sweep=1)) == [1]
    # 多趟：maximal_sweep=2 重复两趟
    @test collect(st.site for st in ZigzagSweep(3; maximal_sweep=2)) ==
          [1, 2, 3, 2, 1, 2, 3, 2]
    @test collect(st.site for st in ZigzagSweep(1; maximal_sweep=3)) == [1, 1, 1]
end

@testset "CircularSweep — 循环与最大趟数" begin
    cs = CircularSweep(4; maximal_sweep = 2)
    @test collect(st.site for st in cs) == [1, 2, 3, 4, 1, 2, 3, 4]
    # 单向循环：dir 恒为 +1
    @test all(st.dir == 1 for st in CircularSweep(4; maximal_sweep = 2))
    # 一趟 N 个 site
    @test length(collect(CircularSweep(3; maximal_sweep = 1))) == 3
    @test length(collect(CircularSweep(1; maximal_sweep = 3))) == 3
end

@testset "advance! — 滑动缓存" begin
    # 前滑：cut 2 更新 site 2 → cut 3
    product = random_periodic_product(3, 5)
    new2 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    product2 = MatrixPeriodicProduct([i == 2 ? new2 : product[i] for i in 1:5])
    v = build_cache(product, 2)
    advance!(v, product2, SweepState(2, 1, 0))
    @test v.site == 3
    @test environment(v) ≈ naive_complement(product2, 3)

    # 后滑：cut 4 更新 site 4 → cut 3
    new4 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    product4 = MatrixPeriodicProduct([i == 4 ? new4 : product[i] for i in 1:5])
    v = build_cache(product, 4)
    advance!(v, product4, SweepState(4, -1, 0))
    @test v.site == 3
    @test environment(v) ≈ naive_complement(product4, 3)

    # 回卷 N→1：更新 site 5 后整个缓存重建
    new5 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    product5 = MatrixPeriodicProduct([i == 5 ? new5 : product[i] for i in 1:5])
    v = build_cache(product, 5)
    advance!(v, product5, SweepState(5, 1, 0))
    @test v.site == 1
    @test environment(v) ≈ naive_complement(product5, 1)

    # 前滑 cut 1 → cut 2（无前缀可乘，直接写入）
    new1 = TensorMap(randn(ComplexF64, 3, 3), ℂ^3, ℂ^3)
    product1 = MatrixPeriodicProduct([i == 1 ? new1 : product[i] for i in 1:5])
    v = build_cache(product, 1)
    advance!(v, product1, SweepState(1, 1, 0))
    @test v.site == 2
    @test environment(v) ≈ naive_complement(product1, 2)
end
