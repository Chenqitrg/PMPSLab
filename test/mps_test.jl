# Tests for src/mps/ — core periodic-MPS types: HalfInt, HalfPeriodicVector,
# PeriodicMPS, the random/one constructors and lift.

using Test
using Random
using PMPSLab:
    PeriodicMPS,
    HalfInt,
    HalfPeriodicVector,
    virtualspace,
    virtualspaces,
    physicalspace,
    physicalspaces,
    one_mps,
    random_cdl,
    random_cdl_mps,
    lift,
    assertconsistency,
    scalartype
using TensorKit
using MPSKit:
    PeriodicVector, left_virtualspace, right_virtualspace, physicalspace, InfiniteMPS
using MPSKit
import HalfIntegers
using HalfIntegers: half

# ==============================================================================
# HalfInt
# ==============================================================================

@testset "HalfInt — ecosystem type and arithmetic" begin
    @test HalfInt === HalfIntegers.HalfInt
    @test HalfInt(3) == 3
    @test HalfInt(3//2) == 3//2
    @test HalfInt(-3//2) == -3//2
    @test HalfInt(3//2) + 2 == HalfInt(7//2)
    @test HalfInt(3//2) - 2 == HalfInt(-1//2)
    @test half(7) == HalfInt(7//2)
    @test sprint(show, HalfInt(7//2)) == "7/2"
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
    @test v[half(3)] == ℂ^2
    @test v[half(5)] == ℂ^3
    @test v[half(7)] == ℂ^4

    # indexing by plain Int delegates to the wrapped elements
    @test v[1] == ℂ^2
    @test v[3] == ℂ^4

    # half-integer rational positions: v[i - 1//2] is the bond before site i
    @test v[1 - 1//2] == ℂ^4     # bond 0 wraps to bond 3 (PeriodicVector semantics)
    @test v[2 - 1//2] == ℂ^2     # bond after site 1
    @test v[3 - 1//2] == ℂ^3     # bond after site 2
    @test v[4 - 1//2] == ℂ^4     # bond after site 3
    @test v[2 + 1//2] == ℂ^3     # bond after site 2
    @test_throws ArgumentError v[HalfInt(2)]

    # iteration
    @test collect(v) == collect(els)
    @test [x for x in v] == collect(els)
end

# ==============================================================================
# PeriodicMPS — container behaviour
# ==============================================================================

@testset "PeriodicMPS — random generation" begin
    first_sample = rand(Xoshiro(1), PeriodicMPS, Float64, 3, 2, 4)
    replayed_sample = rand(Xoshiro(1), PeriodicMPS, Float64, 3, 2, 4)
    two_leg_sample = rand(PeriodicMPS, ComplexF64, 3, (2, 3), 4)

    @test all(
        first_sample[i] == replayed_sample[i] for i in eachindex(first_sample)
    )
    @test length(codomain(two_leg_sample[1])) == 3
    @test codomain(two_leg_sample[1])[2] == ℂ^2
    @test codomain(two_leg_sample[1])[3] == ℂ^3
    @test_throws ArgumentError rand(PeriodicMPS, ComplexF64, 3, (), 4)
end

@testset "PeriodicMPS — container behaviour" begin
    mps = rand(PeriodicMPS, ComplexF64, 4, 2, 3)

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
    mps = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    original = mps[2]

    bad_left = randn(ComplexF64, ℂ^3 ⊗ ℂ^2, ℂ^2)
    @test_throws TensorKit.SpaceMismatch mps[2] = bad_left
    @test mps[2] === original

    bad_right = randn(ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^3)
    @test_throws TensorKit.SpaceMismatch mps[2] = bad_right
    @test mps[2] === original
    @test assertconsistency(mps) === nothing

    single = rand(PeriodicMPS, ComplexF64, 1, 2, 2)
    replacement = randn(ComplexF64, ℂ^3 ⊗ ℂ^2, ℂ^3)
    single[1] = replacement
    @test single[1] === replacement

    old_single = single[1]
    open_tensor = randn(ComplexF64, ℂ^3 ⊗ ℂ^2, ℂ^2)
    @test_throws TensorKit.SpaceMismatch single[1] = open_tensor
    @test single[1] === old_single
end

@testset "PeriodicMPS — copyto! validates before mutation" begin
    dst = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    old_tensors = collect(dst)
    src = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    parent(src)[2] = randn(ComplexF64, ℂ^3 ⊗ ℂ^2, ℂ^2)

    @test_throws TensorKit.SpaceMismatch copyto!(dst, src)
    @test all(dst[i] === old_tensors[i] for i in eachindex(dst))
    @test assertconsistency(dst) === nothing

    nonuniform = rand(
        PeriodicMPS,
        ComplexF64,
        PeriodicVector(fill(ℂ^2, 4)),
        HalfPeriodicVector(PeriodicVector([ℂ^2, ℂ^3, ℂ^4, ℂ^3])),
    )
    @test copyto!(dst, nonuniform) === dst
    @test all(dst[i] === nonuniform[i] for i in eachindex(dst))
    @test assertconsistency(dst) === nothing

    short = rand(PeriodicMPS, ComplexF64, 3, 2, 2)
    copied = collect(dst)
    @test_throws DimensionMismatch copyto!(dst, short)
    @test all(dst[i] === copied[i] for i in eachindex(dst))
end

@testset "PeriodicMPS — functional tensor mapping" begin
    mps = rand(PeriodicMPS, ComplexF64, 4, 2, 3)
    original_objects = collect(mps)
    original_values = copy.(mps)

    mapped = map_tensors(tensor -> 2 * tensor, mps)

    @test Base.isexported(PMPSLab, :map_tensors)
    @test mapped isa PeriodicMPS
    @test mapped !== mps
    @test parent(mapped) !== parent(mps)
    @test all(mapped[site] ≈ 2 * mps[site] for site in eachindex(mps))
    @test all(mps[site] === original_objects[site] for site in eachindex(mps))
    @test all(mps[site] == original_values[site] for site in eachindex(mps))
    @test assertconsistency(mapped) === nothing

    identity_mapped = map_tensors(identity, mps)
    @test parent(identity_mapped) !== parent(mps)
    @test all(identity_mapped[site] === mps[site] for site in eachindex(mps))

    @test_throws TensorKit.SpaceMismatch map_tensors(mps) do tensor
        randn(
            scalartype(tensor),
            left_virtualspace(tensor) ⊗ physicalspace(tensor),
            ℂ^4,
        )
    end
end

@testset "PeriodicMPS — rejects an empty period" begin
    @test_throws ArgumentError rand(PeriodicMPS, ComplexF64, 0, 2, 3)
    @test_throws ArgumentError one_mps(ComplexF64, 0, 2, 3)
    @test_throws ArgumentError rand(PeriodicMPS, ComplexF64, 0, (2, 2), 3)
    @test_throws ArgumentError one_mps(ComplexF64, 0, (2, 2), 3)
end

@testset "PeriodicMPS — space queries" begin
    N, d, D = 4, 2, 3
    mps = rand(PeriodicMPS, ComplexF64, N, d, D)

    @test virtualspace(mps) == ℂ^D
    @test physicalspace(mps) == ℂ^d
    @test scalartype(mps) == ComplexF64
    @test left_virtualspace(mps, 2) == left_virtualspace(mps[2])
    @test right_virtualspace(mps, 2) == right_virtualspace(mps[2])
    @test MPSKit.physicalspace(mps, 2) == physicalspace(mps[2])
    @test collect(left_virtualspace(mps)) == fill(ℂ^D, N)
    @test collect(right_virtualspace(mps)) == fill(ℂ^D, N)

    # HalfInt getindex: bond label → right virtual space of site n = left of site n+1
    @test mps[half(5)] == right_virtualspace(mps[2])
    @test mps[half(5)] == left_virtualspace(mps[3])
    @test_throws ArgumentError mps[HalfInt(2)]

    vs = virtualspaces(mps)
    @test vs isa HalfPeriodicVector
    @test length(vs) == N
    @test vs[half(3)] == right_virtualspace(mps[1])
    @test vs[half(2N + 1)] == right_virtualspace(mps[N])
    @test [vs[half(2i + 1)] for i in 1:N] ==
          [right_virtualspace(mps[i]) for i in 1:N]

    nonuniform = rand(
        PeriodicMPS,
        ComplexF64,
        PeriodicVector(fill(ℂ^d, N)),
        HalfPeriodicVector(PeriodicVector([ℂ^2, ℂ^3, ℂ^4, ℂ^3])),
    )
    nonuniform_vs = virtualspaces(nonuniform)
    @test all(
        nonuniform_vs[half(2i + 1)] == right_virtualspace(nonuniform[i]) ==
            left_virtualspace(nonuniform[i + 1])
        for i in 1:N
    )

    ps = physicalspaces(mps)
    @test ps isa PeriodicVector
    @test length(ps) == N
    @test all(ps[i] == ℂ^d for i in 1:N)
end

@testset "PeriodicMPS — plain-text display" begin
    mps = rand(PeriodicMPS, ComplexF64, 4, 2, 3)
    summary_text = summary(mps)

    @test startswith(summary_text, "4-site PeriodicMPS(ComplexF64, ")
    @test endswith(summary_text, ") with maximal bond dimension 3")

    shown = sprint(show, MIME"text/plain"(), mps; context = (:limit => false,))
    @test startswith(shown, summary_text * ":\n")
    @test occursin("╭─╮\n│ │ ℂ^3", shown)
    @test occursin("│ ├─[4]─ ℂ^2", shown)
    @test occursin("│ ├─[1]─ ℂ^2\n╰─╯", shown)
    @test !occursin("TensorMap", shown)

    compact = sprint(show, MIME"text/plain"(), mps; context = (:compact => true,))
    @test compact == summary_text

    long_mps = rand(PeriodicMPS, ComplexF64, 10, 2, 3)
    limited = sprint(
        show,
        MIME"text/plain"(),
        long_mps;
        context = (:limit => true, :displaysize => (12, 80)),
    )
    @test occursin("│ │ ⋮", limited)
    @test occursin("│ ├─[10]─", limited)
    @test occursin("│ ├─[1]─", limited)
    @test !occursin("│ ├─[8]─", limited)

    two_leg_mps = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 3)
    two_leg_shown = sprint(
        show, MIME"text/plain"(), two_leg_mps; context = (:limit => false,)
    )
    @test occursin("(ℂ^2 ⊗ ℂ^2)", two_leg_shown)
end

# ==============================================================================
# Constructors with PeriodicVector/HalfPeriodicVector spaces
# (the graded analogue of the (d, D) constructors)
# ==============================================================================

@testset "rand / one_mps — PeriodicVector + HalfPeriodicVector spaces" begin
    P = PeriodicVector([ℂ^2, ℂ^2, ℂ^2, ℂ^2])
    V = HalfPeriodicVector(PeriodicVector([ℂ^3, ℂ^3, ℂ^3, ℂ^3]))

    mps = rand(PeriodicMPS, ComplexF64, P, V)
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
    mg = rand(PeriodicMPS, ComplexF64, Pg, Vg)
    @test length(mg) == 3
    @test assertconsistency(mg) === nothing
    @test left_virtualspace(mg[1]) == Z2Space(0 => 2, 1 => 1)

    @test_throws DimensionMismatch rand(
        PeriodicMPS,
        ComplexF64, PeriodicVector(fill(ℂ^2, 3)), HalfPeriodicVector(fill(ℂ^2, 2))
    )
end

@testset "multi-physical-leg rand / one_mps — periodic spaces" begin
    P = PeriodicVector([ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2])
    V = HalfPeriodicVector(PeriodicVector([ℂ^3, ℂ^3]))

    mps = rand(PeriodicMPS, ComplexF64, P, V)
    @test length(mps) == 2
    @test assertconsistency(mps) === nothing
    @test scalartype(mps[1]) == ComplexF64
    @test length(codomain(mps[1])) == 3          # L ⊗ P_a ⊗ P_b
    @test codomain(mps[1])[2] == ℂ^2
    @test codomain(mps[1])[3] == ℂ^2

    mps1 = one_mps(ComplexF64, P, V)
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
    mps = rand(PeriodicMPS, ComplexF64, 3, 2, 2)

    for fold in 1:3
        m = lift(mps, fold)
        @test length(m) == 3 * fold
        @test assertconsistency(m) === nothing
        @test collect(m) == repeat(collect(mps), fold)
    end

    # fold=1 is the identity
    @test collect(lift(mps, 1)) == collect(mps)

    # Works for two-physical-leg tensors as well.
    mps2 = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    m2 = lift(mps2, 2)
    @test length(m2) == 4
    @test assertconsistency(m2) === nothing
end

@testset "lift — Val{Inf} to InfiniteMPS" begin
    mps = rand(PeriodicMPS, ComplexF64, 3, 2, 2)
    inf = lift(mps, Val(Inf))
    @test inf isa MPSKit.InfiniteMPS
    @test length(inf) == 3
    @test collect(physicalspace(inf)) == collect(physicalspaces(mps))
end
