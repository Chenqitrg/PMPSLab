using PMPSLab: SVDInitialization, GaugeInitialization
function run_end_to_end_split(mpsA, initialization; nsweeps::Int=4)
    algorithm = ALSSplit(;
        initialization,
        sweep=CircularSweep(; max_sweeps=nsweeps),
        solver=LUSolve(),
    )
    result, updates = collect_split(mpsA, algorithm)
    mpsB = result.mps
    sites_per_sweep = 2 * length(mpsA)
    costs = getfield.(updates[sites_per_sweep:sites_per_sweep:end], :local_cost)
    return mpsB, updates, costs
end

function test_end_to_end_cost(mpsA, initialization; nsweeps::Int=4)
    mpsB, updates, costs = run_end_to_end_split(mpsA, initialization; nsweeps)

    @test assertconsistency(mpsB) === nothing
    sites_per_sweep = 2 * length(mpsA)
    @test length(updates) == sites_per_sweep * nsweeps
    @test length(costs) == nsweeps
    @test costs == getfield.(updates[sites_per_sweep:sites_per_sweep:end], :local_cost)
    @test all(isfinite, costs)
    @test all(>=(0), costs)
    @test all(diff(costs) .<= 1e-9 .* max.(costs[1:(end - 1)], 1.0))
    @test costs[end] < costs[1]
    @test costs[end] ≈ squared_distance(mpsA, mpsB) rtol=1e-10 atol=1e-10
    return nothing
end

@testset "split — GaugeInitialization end-to-end cost monitoring" begin
    Random.seed!(20260809)
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 3)
    initialization = GaugeInitialization(;
        initialization=SVDInitialization(; trunc=TruncationStrategy(maxrank=2)),
        gauge_fix=ProjectorGaugeFix(; tol=1e-12, maxiter=100),
    )
    test_end_to_end_cost(mpsA, initialization)

    Random.seed!(20260810)
    mpsA_truncated = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 3)
    truncated_ansatz = GaugeInitialization(;
        initialization=SVDInitialization(; trunc=TruncationStrategy(maxrank=2)),
        gauge_fix=ProjectorGaugeFix(
            ; tol=1e-12, maxiter=100,
            trunc=TruncationStrategy(maxrank=2),
        ),
    )
    test_end_to_end_cost(mpsA_truncated, truncated_ansatz)

    Random.seed!(20260811)
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA_z2 = rand(PeriodicMPS, ComplexF64, 2, P ⊗ P, V)
    test_end_to_end_cost(mpsA_z2, truncated_ansatz)
end

@testset "split — CDL filtering changes every corner bond" begin
    Random.seed!(20260812)
    χ = 2
    mpsA = random_cdl_mps(ComplexF64, 2, χ)
    svd_ansatz = SVDInitialization(; trunc=TruncationStrategy())
    gauge_fix = ProjectorGaugeFix(
        ; tol=1e-12, maxiter=100,
        trunc=TruncationStrategy(atol=1e-10),
    )
    periodic_ansatz = GaugeInitialization(; initialization=svd_ansatz, gauge_fix)

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
    mpsB, updates, costs = run_end_to_end_split(mpsA, periodic_ansatz; nsweeps=2)
    @test length(costs) == 2
    @test costs[1] <= initial_cost + 1e-10 * max(initial_cost, 1.0)
    @test all(diff(costs) .<= 1e-9 .* max.(costs[1:(end - 1)], 1.0))
    @test costs[end] ≈ squared_distance(mpsA, mpsB) rtol=1e-10 atol=1e-10
    @test length(updates) == 8
end

@testset "GaugeInitialization — SVD pre-split + projector gauge fixing" begin
    D, d = 2, 2
    B0 = rand(PeriodicMPS, ComplexF64, 4, d, D)
    A = contract_pairs(B0)
    initialization = GaugeInitialization(;
        initialization=SVDInitialization(; trunc=TruncationStrategy()),
        gauge_fix=ProjectorGaugeFix(),
    )
    mpsB = initial_approximation(A, initialization)

    @test length(mpsB) == 4
    @test assertconsistency(mpsB) === nothing
    @test scalartype(mpsB[1]) == ComplexF64
    @test dense(mpsB) ≈ dense(A) rtol=1e-10
end

@testset "GaugeInitialization — Z2-graded spaces" begin
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA = rand(PeriodicMPS, ComplexF64, 2, P ⊗ P, V)
    initialization = GaugeInitialization(;
        initialization=SVDInitialization(; trunc=TruncationStrategy()),
        gauge_fix=ProjectorGaugeFix(),
    )
    mpsB = initial_approximation(mpsA, initialization)

    @test length(mpsB) == 4
    @test assertconsistency(mpsB) === nothing
    @test scalartype(mpsB[1]) == ComplexF64
end
