# Tests for the PeriodicCache of AbstractPeriodicProduct values.
#
# Reference: products are verified against naive left-to-right ordered
# multiplications. Square random matrices make `*` noncommutative, so any
# ordering mistake in the cache shows up as a failing test.

using Test
import PMPSLab
using PMPSLab:
    PeriodicMPS,
    AbstractPeriodicProduct,
    PeriodicTransferMatrix,
    PeriodicCache,
    build_cache,
    prefix_products,
    suffix_products,
    advance!,
    invalidate!,
    rebuild_left!,
    rebuild_right!,
    rebuild!,
    FormalIdentity,
    UnitalElement,
    compose_product,
    StaleCacheError,
    environment,
    left_cache_before,
    range_product,
    right_cache_after,
    edge_space,
    Forward,
    Backward,
    SweepState,
    ZigzagSweep,
    CircularSweep
using TensorKit
using MPSKit: PeriodicVector

# Minimal periodic-product subtype for testing: a vector of square matrices.
struct MatrixPeriodicProduct{T} <: AbstractPeriodicProduct{T}
    sites::Vector{T}
end

Base.length(v::MatrixPeriodicProduct) = length(v.sites)
Base.getindex(v::MatrixPeriodicProduct, i::Int) = v.sites[i]

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
    for N in [2, 3, 5, 8]
        product = random_periodic_product(3, N)
        v = build_cache(product)
        prefixes = prefix_products(product)
        suffixes = suffix_products(product)

        @test v isa PeriodicCache
        @test length(v) == N
        @test lastindex(v) == N
        @test length(v.left) == N
        @test length(v.right) == N
        @test eltype(v.left) == UnitalElement{
            eltype(product),typeof(edge_space(v))
        }
        @test eltype(v.right) == eltype(v.left)
        @test v.left_valid_through == N - 1
        @test v.right_valid_from == 1
        @test v.left[1] isa FormalIdentity
        @test v.right[N] isa FormalIdentity
        @test prefixes[1] isa FormalIdentity
        @test suffixes[N] isa FormalIdentity
        @test scalartype(v.left[2]) == ComplexF64
        @test v.left[2] !== product[1]
        @test v.right[N - 1] !== product[N]

        for k in 1:(N - 1)
            @test prefixes[k + 1] ≈ v.left[k + 1]
            @test v[1:k] ≈ naive_range_product(product, 1:k)
        end

        for k in 2:N
            @test suffixes[k - 1] ≈ v.right[k - 1]
            @test v[k:N] ≈ naive_range_product(product, k:N)
        end
    end
end

@testset "PeriodicCache — display" begin
    cache = build_cache(random_periodic_product(3, 6))

    @test summary(cache) ==
          "6-site PeriodicCache{TensorMap} (left 6/6, right 6/6 valid)"
    @test sprint(show, cache) == summary(cache)

    invalidate!(cache, 3)
    @test summary(cache) ==
          "6-site PeriodicCache{TensorMap} (left 3/6, right 4/6 valid)"
    shown = sprint(
        show,
        MIME"text/plain"(),
        cache;
        context=(:limit => false,),
    )
    @test occursin("╭─╮       ╭─╮", shown)
    @test occursin("│ ○──[6]──⊙ │", shown)
    @test occursin("│ ●──[3]──● │", shown)
    @test occursin("│ ⊙──[1]──○ │", shown)
    @test endswith(shown, "⊙ identity   ● valid   ○ stale")

    compact = sprint(
        show,
        MIME"text/plain"(),
        cache;
        context=(:compact => true,),
    )
    @test compact == summary(cache)

    long_cache = build_cache(random_periodic_product(2, 20))
    limited = sprint(
        show,
        MIME"text/plain"(),
        long_cache;
        context=(:limit => true, :displaysize => (24, 40)),
    )
    @test occursin("⋮", limited)
    @test occursin("│ ●──[20]──⊙ │", limited)
end

@testset "build_cache — direct element generation" begin
    N = 5
    product = random_periodic_product(3, N)
    calls = zeros(Int, N)
    generated = build_cache(
        N,
        site -> begin
            calls[site] += 1
            copy(product[site])
        end,
    )

    @test calls == fill(2, N)
    for site in 1:N-1
        @test generated.left[site + 1] ≈
              naive_range_product(product, 1:site)
        @test generated.right[site] ≈ naive_range_product(product, site + 1:N)
    end
end

@testset "build_cache — rejects an empty product" begin
    @test_throws ArgumentError build_cache(MatrixPeriodicProduct(TensorMap[]))
    @test_throws ArgumentError prefix_products(MatrixPeriodicProduct(TensorMap[]))
    @test_throws ArgumentError suffix_products(MatrixPeriodicProduct(TensorMap[]))
    @test_throws ArgumentError build_cache(0, identity)
end

@testset "PeriodicCache — directional lengths must match" begin
    cache = build_cache(random_periodic_product(3, 4))
    pop!(cache.right)
    @test_throws DimensionMismatch length(cache)
    @test_throws DimensionMismatch lastindex(cache)
end

@testset "range queries — invalid ranges throw" begin
    product = random_periodic_product(3, 5)
    v = build_cache(product)

    @test_throws ArgumentError v[1:5]   # full period
    @test_throws ArgumentError v[2:3]   # interior range
    @test_throws ArgumentError v[2:4]   # interior range
    @test_throws ArgumentError v[3:3]   # interior singleton
    @test_throws ArgumentError v[3:4]   # suffix does not reach N
    @test_throws ArgumentError v[4:4]   # suffix does not reach N
    @test_throws ArgumentError v[2:1]   # empty range

    @test v[1:3] ≈ naive_range_product(product, 1:3)
    @test v[3:5] ≈ naive_range_product(product, 3:5)
    @test v[4:5] ≈ naive_range_product(product, 4:5)
end

@testset "left/right cache and environment — every cut position" begin
    N, D = 5, 3
    product = random_periodic_product(D, N)
    v = build_cache(product)

    for site in 1:N
        if site == 1
            @test left_cache_before(v, site) ==
                  FormalIdentity{eltype(product)}(edge_space(v))
        else
            @test left_cache_before(v, site) ≈
                  naive_range_product(product, 1:(site - 1))
        end
        if site == N
            @test right_cache_after(v, site) ==
                  FormalIdentity{eltype(product)}(edge_space(v))
        else
            @test right_cache_after(v, site) ≈
                  naive_range_product(product, (site + 1):N)
        end

        # The environment is the product of everything except the supplied site,
        # in right-then-left order (noncommutative: order matters).
        @test environment(v, site) ≈ naive_complement(product, site)
        @test environment(v, site) ≈ compose_product(
            right_cache_after(v, site), left_cache_before(v, site)
        )
    end

    @test_throws BoundsError environment(v, 0)
    @test_throws BoundsError environment(v, N + 1)
end

@testset "units and spaces" begin
    D = 3
    product = random_periodic_product(D, 4)
    v2 = build_cache(product)
    @test edge_space(v2) == codomain(product[1])
    @test dim(edge_space(v2)) == D

    # Empty halves return the unit.
    @test left_cache_before(v2, 1) ==
          FormalIdentity{eltype(product)}(edge_space(v2))
    @test right_cache_after(v2, 4) ==
          FormalIdentity{eltype(product)}(edge_space(v2))

    single = random_periodic_product(D, 1)
    single_cache = build_cache(single)
    @test only(single_cache.left) isa FormalIdentity
    @test only(single_cache.right) isa FormalIdentity
    @test environment(single_cache, 1) ==
          FormalIdentity{eltype(single)}(edge_space(single_cache))
end

@testset "Z2Irrep — unit space and environment" begin
    V = Z2Space(0 => 2, 1 => 2)
    product = random_periodic_product(V, 4)
    v = build_cache(product)
    @test edge_space(v) == codomain(product[1])

    for site in 1:4
        @test environment(v, site) ≈ naive_complement(product, site)
    end
end

@testset "Float64 element type" begin
    product = MatrixPeriodicProduct([TensorMap(randn(Float64, 2, 2), ℝ^2, ℝ^2) for _ in 1:4])
    v = build_cache(product)
    @test v[1:1] ≈ product[1]
    @test v[3:4] ≈ naive_range_product(product, 3:4)
    @test environment(v, 2) ≈ naive_complement(product, 2)
end

@testset "PeriodicTransferMatrix integration" begin
    N, d, D = 4, 2, 2
    mps = rand(PeriodicMPS, ComplexF64, N, d, D)
    ptm = PeriodicTransferMatrix(mps)
    v = build_cache(ptm)

    @test length(v) == N
    @test edge_space(v) == edge_space(ptm)
    @test edge_space(v) == codomain(ptm[1])

    # Every representable range matches a direct ordered product.
    @test v[1:1] ≈ range_product(ptm, 1:1)
    @test v[3:4] ≈ range_product(ptm, 3:4)
    @test v[4:4] ≈ range_product(ptm, 4:4)

    @test left_cache_before(v, 2) ≈ range_product(ptm, 1:1)
    @test right_cache_after(v, 2) ≈ range_product(ptm, 3:4)
    @test environment(v, 2) ≈ naive_complement(ptm, 2)

    # The full-period helper agrees with a direct product.
    @test *(ptm) ≈ naive_range_product(ptm, 1:N)
end

@testset "ZigzagSweep — 往返顺序" begin
    # 1, 2, …, N, N-1, …, 2（不重复访问 1）
    @test collect(st.site for st in ZigzagSweep(; site_count=4, max_sweeps=1)) == [1, 2, 3, 4, 3, 2]
    @test collect(st.site for st in ZigzagSweep(; site_count=5, max_sweeps=1)) == [1, 2, 3, 4, 5, 4, 3, 2]
    # 掉头处状态类型由 Forward 变为 Backward
    states = collect(ZigzagSweep(; site_count=4, max_sweeps=1))
    @test map(st -> st isa SweepState{Forward}, states) ==
          [true, true, true, false, false, false]
    # 多趟：max_sweeps=2 重复两趟
    @test collect(st.site for st in ZigzagSweep(; site_count=3, max_sweeps=2)) ==
          [1, 2, 3, 2, 1, 2, 3, 2]
    @test collect(st.site for st in ZigzagSweep(; site_count=2, max_sweeps=2)) ==
          [1, 2, 1, 2]
end

@testset "CircularSweep — 循环与最大趟数" begin
    cs = CircularSweep(; site_count=4, max_sweeps=2)
    @test collect(st.site for st in cs) == [1, 2, 3, 4, 1, 2, 3, 4]
    # 单向循环：状态类型恒为 Forward
    @test all(st isa SweepState{Forward} for st in CircularSweep(; site_count=4, max_sweeps=2))
    # 一趟 N 个 site
    @test length(collect(CircularSweep(; site_count=3, max_sweeps=1))) == 3
    @test length(collect(CircularSweep(; site_count=1, max_sweeps=3))) == 3
end

@testset "sweep constructors validate their configuration" begin
    @test_throws ArgumentError ZigzagSweep(; site_count=0, max_sweeps=1)
    @test_throws ArgumentError ZigzagSweep(; site_count=1, max_sweeps=1)
    @test_throws ArgumentError CircularSweep(; site_count=0, max_sweeps=1)
    for Sweep in (ZigzagSweep, CircularSweep)
        @test_throws ArgumentError Sweep(; site_count=2, max_sweeps=-1)
        @test isempty(collect(Sweep(; site_count=2, max_sweeps=0)))
    end
end

@testset "sweep domain" begin
    @test PMPSLab.sweep_domain(CircularSweep(; site_count=4, max_sweeps=2)) == Base.OneTo(4)
    @test PMPSLab.sweep_domain(ZigzagSweep(; site_count=3, max_sweeps=2)) == Base.OneTo(3)
end
