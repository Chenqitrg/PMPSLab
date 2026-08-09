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
    Random.seed!(20260809)
    mpsA = random_mps2(ComplexF64, 2, 2, 3)
    ansatz = PeriodicGaugeAnsatz(
        SVDAnsatz(TruncationStrategy(maxrank=2)),
        ProjectorCanonical(; tol=1e-12, maxiter=100),
    )
    test_end_to_end_cost(mpsA, ansatz)

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

    @test length(after_dims) == 4
    @test all(after_dims .< before_dims)
    @test after_dims == [1, χ, 1, χ]

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
