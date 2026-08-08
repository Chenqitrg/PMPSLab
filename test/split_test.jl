# Tests for src/split/ — 2-site → 1-site splitting: ansatz, SplitContext,
# the per-site update rule and the full split run.

using Test
using Random
using PMPSLab:
    PeriodicMPS,
    HalfInt,
    random_mps,
    one_mps,
    random_mps2,
    random_cdl_mps,
    pair_contraction,
    dense,
    assertconsistency,
    scalartype,
    SplitContext,
    split,
    update!,
    ZigzagSweep,
    CircularSweep,
    SweepState,
    PeriodicTransferMatrix,
    squared_distance,
    LUSolve,
    CholeskySolve,
    KrylovSolve,
    PeriodicSplit,
    SplitRun,
    SplitSiteCompleted,
    RandomAnsatz,
    OneAnsatz,
    SVDAnsatz,
    PeriodicGaugeAnsatz
import PMPSLab: initial_approximation, split_site, diff_norm,
                site, this_B, other_B, this_A, pa_from_pb, expectation,
                ProjectorCanonical, gaugefix!
using TensorKit
using KrylovKit: GMRES
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace,
              physicalspace, TruncationStrategy

# ==============================================================================
# Ansätze
# ==============================================================================

@testset "public split API" begin
    @test PMPSLab.split === Base.split
    @test all(name -> name in names(PMPSLab), (
        :PeriodicSplit, :RandomAnsatz, :OneAnsatz, :SVDAnsatz,
        :PeriodicGaugeAnsatz,
    ))
    @test SVDAnsatz().truncation isa TruncationStrategy
    @test PeriodicGaugeAnsatz().svd_pre isa SVDAnsatz
end

@testset "RandomAnsatz / OneAnsatz — construction and initial approximation" begin
    N = 2                      # 2-site MPS with N sites → 2N one-site tensors
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA = random_mps2(ComplexF64, N, P, V)

    for ansatz in (RandomAnsatz(V, 2N), OneAnsatz(V, 2N))
        @test length(ansatz.virtual_space) == 2N
        @test ansatz.virtual_space[HalfInt(1)] == V

        mpsB = initial_approximation(mpsA, ansatz)
        @test length(mpsB) == 2N
        @test assertconsistency(mpsB) === nothing
        @test scalartype(mpsB[1]) == ComplexF64
        # every bond carries the ansatz virtual space
        for i in 1:2N
            @test left_virtualspace(mpsB[i]) == V
        end
        # physical spaces are inherited from the two-site tensors, flattened
        @test physicalspace(mpsB[1]) == P
        @test physicalspace(mpsB[2]) == P
    end

    # length mismatch is rejected
    @test_throws DimensionMismatch initial_approximation(mpsA, RandomAnsatz(V, 2N + 1))
end

@testset "split_site — SVD split of one 2-site tensor" begin
    D, d = 3, 2
    T = randn(ComplexF64, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D)
    trunc = TruncationStrategy()
    B1, B2 = split_site(T, trunc)

    # (L ⊗ P₁ ← χ) and (χ ⊗ P₂ ← R)
    @test length(codomain(B1)) == 2
    @test codomain(B1)[1] == ℂ^D
    @test codomain(B1)[2] == ℂ^d
    @test length(domain(B1)) == 1
    @test length(codomain(B2)) == 2
    @test codomain(B2)[1] == domain(B1)[1]      # shared bond χ
    @test codomain(B2)[2] == ℂ^d
    @test domain(B2)[1] == ℂ^D

    # the pair reconstructs the two-site tensor
    @plansor recon[-1 -2 -3; -4] := B1[-1 -2; 1] * B2[1 -3; -4]
    @test recon ≈ T rtol=1e-12
end

@testset "SVDAnsatz — initial approximation preserves periodic density" begin
    B0 = random_mps(ComplexF64, 4, 2, 2)
    A = pair_contraction(B0)
    mpsB = initial_approximation(A, SVDAnsatz(TruncationStrategy()))

    @test length(mpsB) == 4
    @test assertconsistency(mpsB) === nothing
    # exact for factorizable A (the SVD rank is not truncated)
    @test dense(mpsB) ≈ dense(A) rtol=1e-10
end

# ==============================================================================
# SplitContext
# ==============================================================================

@testset "SplitContext — construction and accessors" begin
    D, d = 2, 2
    mpsB = random_mps(ComplexF64, 4, d, D)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, d, D))
    ctx = SplitContext(mpsA, mpsB)

    @test site(ctx) == 1
    @test this_B(ctx) == mpsB[1]
    @test other_B(ctx) == mpsB[2]           # odd site: the right neighbour
    @test this_A(ctx) == mpsA[1]
    @test pa_from_pb(1) == 1 && pa_from_pb(2) == 1 && pa_from_pb(3) == 2

    # the caches sit at cut 1, the A-pair expectation is precomputed
    @test ctx.cacheBB.site == 1
    @test ctx.cacheAB.site == 1
    @test ctx.expectationAA ≈ expectation(ctx.braketAA)

    # mismatched site counts are rejected
    @test_throws DimensionMismatch SplitContext(mpsA, random_mps(ComplexF64, 3, d, D))
end

@testset "update! — Krylov solver follows the split protocol" begin
    mpsA = random_mps2(ComplexF64, 2, 2, 1)
    mpsB = initial_approximation(mpsA, SVDAnsatz())
    ctx = SplitContext(mpsA, mpsB)
    solver = KrylovSolve(GMRES(; maxiter=100, tol=1e-12))

    _, cost = update!(ctx, solver, SweepState(1, 1, 0))
    @test isfinite(cost)
    @test ctx.cacheBB.site == 2
end

@testset "update! — site mismatch throws" begin
    mpsB = random_mps(ComplexF64, 4, 2, 2)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, 2, 2))
    ctx = SplitContext(mpsA, mpsB)
    @test_throws ArgumentError update!(ctx, CholeskySolve(), SweepState(2, 1, 0))
end

@testset "update! — one forward step slides both caches" begin
    D, d = 2, 2
    mpsB = random_mps(ComplexF64, 4, d, D)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, d, D))
    ctx = SplitContext(mpsA, mpsB)
    old1 = copy(mpsB[1])

    # step 1: update site 1 (odd) — only the BB cache slides
    update!(ctx, CholeskySolve(), SweepState(1, 1, 0))
    @test site(ctx) == 2
    @test !(ctx.mpsB[1] ≈ old1)
    @test ctx.cacheBB.site == 2
    @test ctx.cacheAB.site == 1          # AB cache only moves at even sites

    # step 2: update site 2 (even) — the AB cache slides too
    update!(ctx, CholeskySolve(), SweepState(2, 1, 0))
    @test site(ctx) == 3
    @test ctx.cacheBB.site == 3
    @test ctx.cacheAB.site == 2

    # the new tensor keeps the site's space
    @test space(ctx.mpsB[2]) == space(mpsB[2])
end

@testset "update! — one step recovers exact B from perturbed guess" begin
    B0 = random_mps(ComplexF64, 4, 2, 2)
    A = pair_contraction(B0)

    # odd site: perturb site 1, others exact → one update recovers B0[1]
    B_init = deepcopy(B0)
    B_init[1] = rand(ComplexF64, space(B0[1]))
    ctx = SplitContext(A, B_init)
    update!(ctx, CholeskySolve(), SweepState(1, 1, 0))
    @test ctx.mpsB[1] ≈ B0[1] rtol=1e-10
end

# ==============================================================================
# split — full run
# ==============================================================================

@testset "SplitRun — one safe step updates exactly one site" begin
    mpsA = random_mps2(ComplexF64, 2, 2, 1)
    algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(4; maximal_sweep=1),
        CholeskySolve(),
    )
    run = initialize_run(mpsA, algorithm)

    @test run isa SplitRun
    @test run.context isa SplitContext
    @test !hasproperty(run, :target)
    @test !isfinished(run)
    @test first(run.pending_iteration) == SweepState(1, 1, 0)
    @test last(run.pending_iteration) == SweepState(2, 1, 0)
    @test isempty(run.log.site_logs)
    @test_throws ArgumentError run_result(run)

    event = step!(run)
    @test event isa SplitSiteCompleted
    @test event.log.state == SweepState(1, 1, 0)
    @test length(run.log.site_logs) == 1
    @test site(run.context) == 2
    @test first(run.pending_iteration) == SweepState(2, 1, 0)
    @test !isfinished(run)

    while !isfinished(run)
        step!(run)
    end
    mpsB, log = run_result(run)
    @test mpsB === run.context.mpsB
    @test log === run.log
    @test length(log.site_logs) == 4
    @test log.sweep_time == sum(site_log -> site_log.time, log.site_logs)
    @test_throws ArgumentError step!(run)

    blocking_mps, blocking_log = split(mpsA, algorithm)
    @test dense(blocking_mps) ≈ dense(mpsB)
    @test getfield.(blocking_log.site_logs, :error) ≈
          getfield.(log.site_logs, :error)
    @test_throws MethodError split(mpsA, algorithm; callback=nothing)
end

@testset "SplitRun — an empty sweep is complete at initialization" begin
    mpsA = random_mps2(ComplexF64, 2, 2, 1)
    algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(4; maximal_sweep=0),
        CholeskySolve(),
    )
    run = initialize_run(mpsA, algorithm)

    @test isfinished(run)
    @test isnothing(run.pending_iteration)
    @test isempty(last(run_result(run)).site_logs)
end

@testset "SplitSnapshot — captures an isolated safe-boundary state" begin
    mpsA = random_mps2(ComplexF64, 2, 2, 1)
    algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(4; maximal_sweep=1),
        CholeskySolve(),
    )
    run = initialize_run(mpsA, algorithm)
    step!(run)

    saved = snapshot(run)
    @test saved isa SplitSnapshot
    @test saved.mpsB !== run.context.mpsB
    @test parent(saved.mpsB) !== parent(run.context.mpsB)
    @test all(saved.mpsB[i] !== run.context.mpsB[i] for i in eachindex(saved.mpsB))
    @test saved.pending_iteration == run.pending_iteration
    @test saved.log !== run.log
    @test saved.log.site_logs !== run.log.site_logs
    @test saved.log.site_logs == run.log.site_logs
    @test saved.log.sweep_time == run.log.sweep_time

    saved_next_tensor = copy(saved.mpsB[2])
    step!(run)
    @test saved.pending_iteration != run.pending_iteration
    @test length(saved.log.site_logs) == 1
    @test length(run.log.site_logs) == 2
    @test saved.mpsB[2] == saved_next_tensor

    while !isfinished(run)
        step!(run)
    end
    completed = snapshot(run)
    @test isnothing(completed.pending_iteration)
    @test length(completed.log.site_logs) == 4
end

@testset "split — SVDAnsatz, circular sweep, exact solve" begin
    B0 = random_mps(ComplexF64, 4, 2, 2)
    A = pair_contraction(B0)
    alg = PeriodicSplit(
        SVDAnsatz(TruncationStrategy()),
        CircularSweep(4; maximal_sweep=1),
        CholeskySolve(),
    )
    B1, log = split(A, alg)
    
    @test length(B1) == 4
    @test assertconsistency(B1) === nothing
    @test scalartype(B1[1]) == ComplexF64
    # the periodic state is preserved: dense(B) ≈ dense(A)
    @test dense(B1) ≈ dense(A) rtol=1e-6
end

@testset "split — exact B0 stays unchanged after full sweep" begin
    B0 = random_mps(ComplexF64, 4, 2, 2)
    A = pair_contraction(B0)
    ctx = SplitContext(A, deepcopy(B0))
    for st in CircularSweep(4; maximal_sweep=1)
        update!(ctx, CholeskySolve(), st)
    end
    @test diff_norm(ctx.mpsB, B0) ≈ 0 atol=1e-10
end

@testset "split — RandomAnsatz + Z2 spaces, zigzag sweep" begin
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA = random_mps2(ComplexF64, 2, P, V)
    alg = PeriodicSplit(
        RandomAnsatz(V, 4),
        ZigzagSweep(4; maximal_sweep=1),
        LUSolve(),
    )
    B1, log = split(mpsA, alg)

    @test length(B1) == 4
    @test assertconsistency(B1) === nothing
    @test scalartype(B1[1]) == ComplexF64
end

@testset "update! — exact periodic cost descends for both sweep orders" begin
    Random.seed!(1234)
    mpsA = random_mps2(ComplexF64, 2, 2, 2)
    ansatz = SVDAnsatz(TruncationStrategy(maxrank=1))

    for sweep in (
            CircularSweep(4; maximal_sweep=2),
            ZigzagSweep(4; maximal_sweep=2),
        )
        ctx = SplitContext(mpsA, initial_approximation(mpsA, ansatz))
        previous_cost = squared_distance(mpsA, ctx.mpsB)

        for st in sweep
            _, local_cost = update!(ctx, LUSolve(), st)
            exact_cost = squared_distance(mpsA, ctx.mpsB)
            @test local_cost ≈ exact_cost rtol=1e-9 atol=1e-9
            @test exact_cost <= previous_cost + 1e-9 * max(previous_cost, 1.0)
            previous_cost = exact_cost
        end
    end
end

@testset "split — sweep progress log" begin
    Random.seed!(4321)
    mpsA = random_mps2(ComplexF64, 2, 2, 2)
    nsweeps = 3
    alg = PeriodicSplit(
        SVDAnsatz(TruncationStrategy(maxrank=1)),
        CircularSweep(4; maximal_sweep=nsweeps),
        LUSolve(),
    )
    mpsB, log = split(mpsA, alg)
    sweep_costs = getfield.(log.site_logs[4:4:end], :error)

    @test length(log.site_logs) == 4nsweeps
    @test length(sweep_costs) == nsweeps
    @test all(diff(sweep_costs) .<= 1e-9 .* max.(sweep_costs[1:(end - 1)], 1.0))
    @test sweep_costs[end] ≈ squared_distance(mpsA, mpsB) rtol=1e-10 atol=1e-10
    @test log.sweep_time >= 0
end

function run_end_to_end_split(mpsA, ansatz; nsweeps::Int=4)
    alg = PeriodicSplit(
        ansatz,
        CircularSweep(2 * length(mpsA); maximal_sweep=nsweeps),
        LUSolve(),
    )
    mpsB, log = split(mpsA, alg)
    sites_per_sweep = 2 * length(mpsA)
    costs = getfield.(log.site_logs[sites_per_sweep:sites_per_sweep:end], :error)
    return mpsB, log, costs
end

function test_end_to_end_cost(mpsA, ansatz; nsweeps::Int=4)
    mpsB, log, costs = run_end_to_end_split(mpsA, ansatz; nsweeps)

    @test assertconsistency(mpsB) === nothing
    sites_per_sweep = 2 * length(mpsA)
    @test length(log.site_logs) == sites_per_sweep * nsweeps
    @test length(costs) == nsweeps
    @test costs == getfield.(log.site_logs[sites_per_sweep:sites_per_sweep:end], :error)
    @test all(isfinite, costs)
    @test all(>=(0), costs)
    @test all(diff(costs) .<= 1e-9 .* max.(costs[1:(end - 1)], 1.0))
    @test costs[end] < costs[1]
    @test costs[end] ≈ squared_distance(mpsA, mpsB) rtol=1e-10 atol=1e-10
    return nothing
end

@testset "split — PeriodicGaugeAnsatz end-to-end cost monitoring" begin
    # Local SVD truncation supplies a nonzero initial error for ALS to reduce;
    # projector filtering is otherwise exact in this case.
    Random.seed!(20260809)
    mpsA = random_mps2(ComplexF64, 2, 2, 3)
    ansatz = PeriodicGaugeAnsatz(
        SVDAnsatz(TruncationStrategy(maxrank=2)),
        ProjectorCanonical(; tol=1e-12, maxiter=100),
    )
    test_end_to_end_cost(mpsA, ansatz)

    # Exercise nontrivial projector compression in addition to the local SVD.
    Random.seed!(20260810)
    mpsA_truncated = random_mps2(ComplexF64, 2, 2, 3)
    truncated_ansatz = PeriodicGaugeAnsatz(
        SVDAnsatz(TruncationStrategy(maxrank=2)),
        ProjectorCanonical(
            ; tol=1e-12, maxiter=100,
            trscheme=TruncationStrategy(maxrank=2),
        ),
    )
    test_end_to_end_cost(mpsA_truncated, truncated_ansatz)

    # The same compressed workflow must preserve graded-space consistency.
    Random.seed!(20260811)
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA_z2 = random_mps2(ComplexF64, 2, P, V)
    test_end_to_end_cost(mpsA_z2, truncated_ansatz)
end

@testset "split — CDL filtering changes every corner bond" begin
    Random.seed!(20260812)
    χ = 2
    mpsA = random_cdl_mps(ComplexF64, 2, χ)
    svd_ansatz = SVDAnsatz(TruncationStrategy())
    canonical = ProjectorCanonical(
        ; tol=1e-12, maxiter=100,
        trscheme=TruncationStrategy(atol=1e-10),
    )
    periodic_ansatz = PeriodicGaugeAnsatz(svd_ansatz, canonical)

    before_filtering = initial_approximation(mpsA, svd_ansatz)
    after_filtering = initial_approximation(mpsA, periodic_ansatz)
    before_dims = [dim(right_virtualspace(tensor)) for tensor in before_filtering]
    after_dims = [dim(right_virtualspace(tensor)) for tensor in after_filtering]

    # The local SVD exposes four corner bonds. Projector filtering acts on all
    # four of them, rather than only changing one reference cut.
    @test length(after_dims) == 4
    @test all(after_dims .< before_dims)
    @test after_dims == [1, χ, 1, χ]

    # Filtering removes the closed double line. It can change its global scalar,
    # but not the projective physical state.
    target = dense(mpsA)
    filtered = dense(after_filtering)
    α = dot(target, filtered) / dot(target, target)
    @test norm(filtered - α * target) / norm(filtered) < 1e-12

    initial_cost = squared_distance(mpsA, after_filtering)
    mpsB, log, costs = run_end_to_end_split(mpsA, periodic_ansatz; nsweeps=2)
    @test length(costs) == 2
    @test costs[1] <= initial_cost + 1e-10 * max(initial_cost, 1.0)
    @test all(diff(costs) .<= 1e-9 .* max.(costs[1:(end - 1)], 1.0))
    @test costs[end] ≈ squared_distance(mpsA, mpsB) rtol=1e-10 atol=1e-10
    @test length(log.site_logs) == 8
end

@testset "PeriodicGaugeAnsatz — SVD pre-split + projector gauge fixing" begin
    D, d = 2, 2
    B0 = random_mps(ComplexF64, 4, d, D)
    A = pair_contraction(B0)
    ansatz = PeriodicGaugeAnsatz(SVDAnsatz(TruncationStrategy()), ProjectorCanonical())
    mpsB = initial_approximation(A, ansatz)

    @test length(mpsB) == 4
    @test assertconsistency(mpsB) === nothing
    @test scalartype(mpsB[1]) == ComplexF64
    # SVD is exact for factorizable A, and gaugefix! doesn't change the state.
    @test dense(mpsB) ≈ dense(A) rtol=1e-10
end

@testset "PeriodicGaugeAnsatz — Z2-graded spaces" begin
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA = random_mps2(ComplexF64, 2, P, V)
    ansatz = PeriodicGaugeAnsatz(SVDAnsatz(TruncationStrategy()), ProjectorCanonical())
    mpsB = initial_approximation(mpsA, ansatz)

    @test length(mpsB) == 4
    @test assertconsistency(mpsB) === nothing
    @test scalartype(mpsB[1]) == ComplexF64
end
