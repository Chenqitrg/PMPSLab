# Tests for src/mps/ — core periodic-MPS types: HalfInt, HalfPeriodicVector,
# PeriodicMPS, the random/one constructors and lift.

using Test
using PMPSLab:
    PeriodicMPS,
    HalfInt,
    HalfPeriodicVector,
    virtualspace,
    virtualspaces,
    physicalspace,
    physicalspaces,
    random_mps,
    one_mps,
    random_mps2,
    one_mps2,
    random_cdl,
    random_cdl_mps,
    lift,
    assertconsistency,
    scalartype
using TensorKit
using MPSKit:
    PeriodicVector, left_virtualspace, right_virtualspace, physicalspace, InfiniteMPS
using MPSKit

# ==============================================================================
# HalfInt
# ==============================================================================

@testset "HalfInt — Int constructor and arithmetic" begin
    h = HalfInt(3)
    @test h.n == 3
    @test HalfInt(0).n == 0
    @test HalfInt(-2).n == -2

    # == and hash follow the wrapped Int
    @test HalfInt(3) == HalfInt(3)
    @test HalfInt(3) != HalfInt(4)
    @test hash(HalfInt(3)) == hash(HalfInt(3))

    # convert from Int
    @test convert(HalfInt, 7) == HalfInt(7)

    # arithmetic with Int
    @test (HalfInt(3) + 2) == HalfInt(5)
    @test (HalfInt(3) - 2) == HalfInt(1)
end

@testset "HalfInt — Rational constructor (n + 1/2)" begin
    # exact half-integers: HalfInt(x) with x = n + 1/2
    @test HalfInt(1//2).n == 0      # 0 + 1/2
    @test HalfInt(3//2).n == 1      # 1 + 1/2
    @test HalfInt(5//2).n == 2      # 2 + 1/2
    @test HalfInt(-1//2).n == -1    # -1 + 1/2
    @test HalfInt(-3//2).n == -2

    # a pure integer is not of the form n + 1/2 → must be rejected
    @test_throws ArgumentError HalfInt(2//1)
    @test_throws ArgumentError HalfInt(0//1)
    @test_throws ArgumentError HalfInt(-5//1)

    # denominators other than 2 are rejected
    @test_throws ArgumentError HalfInt(1//3)
    @test_throws ArgumentError HalfInt(4//7)
end

# ==============================================================================
# HalfPeriodicVector
# ==============================================================================

@testset "HalfPeriodicVector — indexing and iteration" begin
    els = PeriodicVector([ℂ^2, ℂ^3, ℂ^4])
    v = HalfPeriodicVector(els)

    @test v isa AbstractVector
    @test length(v) == 3
    @test size(v) == (3,)
    @test axes(v) == (Base.OneTo(3),)
    @test eachindex(v) == Base.OneTo(3)
    @test parent(v) === els
    @test eltype(typeof(v)) == ComplexSpace

    # indexing by HalfInt bond label
    @test v[HalfInt(1)] == ℂ^2
    @test v[HalfInt(2)] == ℂ^3
    @test v[HalfInt(3)] == ℂ^4

    # indexing by plain Int delegates to the wrapped elements
    @test v[1] == ℂ^2
    @test v[3] == ℂ^4

    # half-integer rational positions: v[i - 1//2] is the bond before site i
    @test v[1 - 1//2] == ℂ^4     # bond 0 wraps to bond 3 (PeriodicVector semantics)
    @test v[2 - 1//2] == ℂ^2     # same as v[HalfInt(1)]
    @test v[3 - 1//2] == ℂ^3     # same as v[HalfInt(2)]
    @test v[4 - 1//2] == ℂ^4     # same as v[HalfInt(3)]
    @test v[2 + 1//2] == ℂ^3     # same as v[HalfInt(2)]

    # iteration
    @test collect(v) == collect(els)
    @test [x for x in v] == collect(els)
end

# ==============================================================================
# PeriodicMPS — container behaviour
# ==============================================================================

@testset "PeriodicMPS — container behaviour" begin
    mps = random_mps(ComplexF64, 4, 2, 3)

    @test length(mps) == 4
    @test firstindex(mps) == 1
    @test lastindex(mps) == 4
    @test eachindex(mps) == Base.OneTo(4)
    @test parent(mps) === mps.tensors
    @test eltype(typeof(mps)) == typeof(mps[1])
    @test mps[1] == mps[1]
    @test mps[0] == mps[4]                      # PeriodicVector wraps around
    @test mps[5] == mps[1]

    # setindex!
    new = randn(ComplexF64, ℂ^3 ⊗ ℂ^2, ℂ^3)
    mps[2] = new
    @test mps[2] == new

    # iteration
    @test collect(mps) == collect(mps.tensors)

    copied = similar(mps)
    @test copyto!(copied, mps) === copied
    @test all(copied[i] == mps[i] for i in eachindex(mps))
end

@testset "PeriodicMPS — rejected writes are atomic" begin
    mps = random_mps(ComplexF64, 4, 2, 2)
    original = mps[2]

    bad_left = randn(ComplexF64, ℂ^3 ⊗ ℂ^2, ℂ^2)
    @test_throws TensorKit.SpaceMismatch mps[2] = bad_left
    @test mps[2] === original

    bad_right = randn(ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^3)
    @test_throws TensorKit.SpaceMismatch mps[2] = bad_right
    @test mps[2] === original
    @test assertconsistency(mps) === nothing

    single = random_mps(ComplexF64, 1, 2, 2)
    replacement = randn(ComplexF64, ℂ^3 ⊗ ℂ^2, ℂ^3)
    single[1] = replacement
    @test single[1] === replacement

    old_single = single[1]
    open_tensor = randn(ComplexF64, ℂ^3 ⊗ ℂ^2, ℂ^2)
    @test_throws TensorKit.SpaceMismatch single[1] = open_tensor
    @test single[1] === old_single
end

@testset "PeriodicMPS — copyto! validates before mutation" begin
    dst = random_mps(ComplexF64, 4, 2, 2)
    old_tensors = collect(dst)
    src = random_mps(ComplexF64, 4, 2, 2)
    parent(src)[2] = randn(ComplexF64, ℂ^3 ⊗ ℂ^2, ℂ^2)

    @test_throws TensorKit.SpaceMismatch copyto!(dst, src)
    @test all(dst[i] === old_tensors[i] for i in eachindex(dst))
    @test assertconsistency(dst) === nothing

    nonuniform = random_mps(
        ComplexF64,
        PeriodicVector(fill(ℂ^2, 4)),
        HalfPeriodicVector(PeriodicVector([ℂ^2, ℂ^3, ℂ^4, ℂ^3])),
    )
    @test copyto!(dst, nonuniform) === dst
    @test all(dst[i] === nonuniform[i] for i in eachindex(dst))
    @test assertconsistency(dst) === nothing

    short = random_mps(ComplexF64, 3, 2, 2)
    copied = collect(dst)
    @test_throws DimensionMismatch copyto!(dst, short)
    @test all(dst[i] === copied[i] for i in eachindex(dst))
end

@testset "PeriodicMPS — rejects an empty period" begin
    @test_throws ArgumentError random_mps(ComplexF64, 0, 2, 3)
    @test_throws ArgumentError one_mps(ComplexF64, 0, 2, 3)
    @test_throws ArgumentError random_mps2(ComplexF64, 0, 2, 3)
    @test_throws ArgumentError one_mps2(ComplexF64, 0, 2, 3)
end

@testset "PeriodicMPS — space queries" begin
    N, d, D = 4, 2, 3
    mps = random_mps(ComplexF64, N, d, D)

    @test virtualspace(mps) == ℂ^D
    @test physicalspace(mps) == ℂ^d
    @test scalartype(mps) == ComplexF64

    # HalfInt getindex: bond label → right virtual space of site n = left of site n+1
    @test mps[HalfInt(2)] == right_virtualspace(mps[2])
    @test mps[HalfInt(2)] == left_virtualspace(mps[3])

    vs = virtualspaces(mps)
    @test vs isa HalfPeriodicVector
    @test length(vs) == N
    @test vs[HalfInt(1)] == right_virtualspace(mps[1])
    @test vs[HalfInt(N)] == right_virtualspace(mps[N])
    @test [vs[HalfInt(i)] for i in 1:N] == [right_virtualspace(mps[i]) for i in 1:N]

    nonuniform = random_mps(
        ComplexF64,
        PeriodicVector(fill(ℂ^d, N)),
        HalfPeriodicVector(PeriodicVector([ℂ^2, ℂ^3, ℂ^4, ℂ^3])),
    )
    nonuniform_vs = virtualspaces(nonuniform)
    @test all(
        nonuniform_vs[HalfInt(i)] == right_virtualspace(nonuniform[i]) ==
            left_virtualspace(nonuniform[i + 1])
        for i in 1:N
    )

    ps = physicalspaces(mps)
    @test ps isa PeriodicVector
    @test length(ps) == N
    @test all(ps[i] == ℂ^d for i in 1:N)
end

# ==============================================================================
# Constructors with PeriodicVector/HalfPeriodicVector spaces
# (the graded analogue of the (d, D) constructors)
# ==============================================================================

@testset "random_mps / one_mps — PeriodicVector + HalfPeriodicVector spaces" begin
    P = PeriodicVector([ℂ^2, ℂ^2, ℂ^2, ℂ^2])
    V = HalfPeriodicVector(PeriodicVector([ℂ^3, ℂ^3, ℂ^3, ℂ^3]))

    mps = random_mps(ComplexF64, P, V)
    @test length(mps) == 4
    @test assertconsistency(mps) === nothing
    @test scalartype(mps[1]) == ComplexF64
    @test left_virtualspace(mps[1]) == ℂ^3
    @test physicalspace(mps[1]) == ℂ^2

    mps1 = one_mps(ComplexF64, P, V)
    @test length(mps1) == 4
    @test assertconsistency(mps1) === nothing
    @test all(all(mps1[i][a, b, c] == 1 for a in 1:3, b in 1:2, c in 1:3) for i in 1:4)

    # Z2-graded spaces
    Pg = PeriodicVector([Z2Space(0 => 2), Z2Space(0 => 2), Z2Space(0 => 2)])
    Vg = HalfPeriodicVector(PeriodicVector([Z2Space(0 => 2, 1 => 1), Z2Space(0 => 2, 1 => 1), Z2Space(0 => 2, 1 => 1)]))
    mg = random_mps(ComplexF64, Pg, Vg)
    @test length(mg) == 3
    @test assertconsistency(mg) === nothing
    @test left_virtualspace(mg[1]) == Z2Space(0 => 2, 1 => 1)

    @test_throws DimensionMismatch random_mps(
        ComplexF64, PeriodicVector(fill(ℂ^2, 3)), HalfPeriodicVector(fill(ℂ^2, 2))
    )
end

@testset "random_mps2 / one_mps2 — PeriodicVector + HalfPeriodicVector spaces" begin
    P = PeriodicVector([ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2])
    V = HalfPeriodicVector(PeriodicVector([ℂ^3, ℂ^3]))

    mps = random_mps2(ComplexF64, P, V)
    @test length(mps) == 2
    @test assertconsistency(mps) === nothing
    @test scalartype(mps[1]) == ComplexF64
    @test length(codomain(mps[1])) == 3          # L ⊗ P_a ⊗ P_b
    @test codomain(mps[1])[2] == ℂ^2
    @test codomain(mps[1])[3] == ℂ^2

    mps1 = one_mps2(ComplexF64, P, V)
    @test length(mps1) == 2
    @test assertconsistency(mps1) === nothing
    @test all(all(mps1[i][a, b, c, d] == 1 for a in 1:3, b in 1:2, c in 1:2, d in 1:3) for i in 1:2)
end

@testset "random_cdl — prescribed six-leg layout" begin
    Vdl, Vdr = ℂ^2, ℂ^3
    Vlu, Vld = ℂ^4, ℂ^5
    Vru, Vrd = ℂ^4, ℂ^3
    tensor = random_cdl(Vdl, Vdr, Vlu, Vld, Vru, Vrd)

    @test space(tensor) ==
          (Vlu ⊗ Vld' ⊗ Vdl ⊗ Vdr' ← Vru ⊗ Vrd')
    @test scalartype(tensor) == ComplexF64
end

@testset "random_cdl_mps — inverse fusion unitaries" begin
    N, χ = 4, 2
    mps = random_cdl_mps(ComplexF64, N, χ)

    @test length(mps) == N
    @test assertconsistency(mps) === nothing
    @test all(dim(left_virtualspace(tensor)) == χ^2 for tensor in mps)
    @test all(dim(right_virtualspace(tensor)) == χ^2 for tensor in mps)
    @test all(numout(tensor) == 3 && numin(tensor) == 1 for tensor in mps)
end

# ==============================================================================
# lift — replicate the unit cell
# ==============================================================================

@testset "lift — finite fold" begin
    mps = random_mps(ComplexF64, 3, 2, 2)

    for fold in 1:3
        m = lift(mps, fold)
        @test length(m) == 3 * fold
        @test assertconsistency(m) === nothing
        @test collect(m) == repeat(collect(mps), fold)
    end

    # fold=1 is the identity
    @test collect(lift(mps, 1)) == collect(mps)

    # works for 2-site MPS as well
    mps2 = random_mps2(ComplexF64, 2, 2, 2)
    m2 = lift(mps2, 2)
    @test length(m2) == 4
    @test assertconsistency(m2) === nothing
end

@testset "lift — Val{Inf} to InfiniteMPS" begin
    mps = random_mps(ComplexF64, 3, 2, 2)
    inf = lift(mps, Val(Inf))
    @test inf isa MPSKit.InfiniteMPS
    @test length(inf) == 3
    @test collect(physicalspace(inf)) == collect(physicalspaces(mps))
end
