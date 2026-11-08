using PMPSLab: RandomInitialization, IdentityInitialization, SVDInitialization, GaugeInitialization, SplitEnvironmentPolicy, CostTolerance, CostConvergence
using TensorKit: MatrixAlgebraKit
@testset "algorithm configurations have value semantics" begin
    virtual_space = Z2Space(0 => 2, 1 => 1)
    configurations = (
        RandomInitialization(; virtual_spaces=fill(virtual_space, 4)),
        IdentityInitialization(; virtual_spaces=fill(virtual_space, 4)),
        SVDInitialization(; trunc=TruncationStrategy()),
        GaugeInitialization(;
            initialization=SVDInitialization(; trunc=TruncationStrategy()),
            gauge_fix=ProjectorGaugeFix(),
        ),
        CircularSweep(; max_sweeps=2),
        ZigzagSweep(; max_sweeps=2),
        LUSolve(),
        CholeskySolve(),
        KrylovSolve(; algorithm=GMRES(; maxiter=10)),
        CostTolerance(; atol=1e-9, rtol=1e-2, min_sweeps=5),
        CostConvergence(),
        SplitEnvironmentPolicy(),
        SplitEnvironmentPolicy(;
            normal=TruncatedTransferPolicy(;
                trunc=TruncationStrategy()
            ),
            mixed=TruncatedTransferPolicy(;
                trunc=TruncationStrategy()
            ),
        ),
    )

    for configuration in configurations
        copied = deepcopy(configuration)
        @test configuration isa ExperimentLab.AbstractAlgorithmConfig
        @test configuration == copied
        @test isequal(configuration, copied)
        @test hash(configuration) == hash(copied)
    end

    algorithm = ALSSplit(;
        initialization=first(configurations),
        sweep=CircularSweep(; max_sweeps=1),
        solver=LUSolve(),
    )
    @test algorithm == deepcopy(algorithm)
    @test hash(algorithm) == hash(deepcopy(algorithm))
end

@testset "algorithm configurations — editable automatic display" begin
    initialization = GaugeInitialization()
    ansatz_text = sprint(show, MIME"text/plain"(), initialization)
    @test startswith(ansatz_text, "PMPSLab.GaugeInitialization\n")
    @test occursin("initialization: PMPSLab.SVDInitialization", ansatz_text)
    @test occursin("gauge_fix: PMPSLab.ProjectorGaugeFix", ansatz_text)
    @test occursin("MatrixAlgebraKit.notrunc()", ansatz_text)
    @test isequal(include_string(@__MODULE__, "@config " * repr(ansatz_text)), initialization)
    @test isequal(include_string(@__MODULE__, sprint(show, initialization)), initialization)
    @test isequal(include_string(@__MODULE__, sprint(show, LUSolve())), LUSolve())
    @test sprint(show, MIME"text/plain"(), initialization; context=:compact=>true) == "GaugeInitialization(...)"

    split_algorithm = ALSSplit(;
        initialization,
        sweep=CircularSweep(; max_sweeps=1),
        solver=CholeskySolve(),
    )
    split_text = sprint(show, MIME"text/plain"(), split_algorithm)
    @test occursin("solver: TensorLinearSolveLab.CholeskySolve()", split_text)
    @test occursin("normal: PMPSLab.DenseTransferPolicy()", split_text)
    @test occursin("stopping: PMPSLab.NoStopping()", split_text)
    @test isequal(include_string(@__MODULE__, "@config " * repr(split_text)), split_algorithm)

    uncommon_truncation = SVDInitialization(; trunc=truncrank(2) | truncerror(; rtol=1e-8))
    shown = sprint(show, MIME"text/plain"(), uncommon_truncation)
    @test occursin("Display only", shown)
    @test occursin("trunc: TruncationUnion(...)", shown)
end

@testset "split — SVDInitialization, circular sweep, exact solve" begin
    B0 = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    A = contract_pairs(B0)
    algorithm = ALSSplit(;
        initialization=SVDInitialization(; trunc=TruncationStrategy()),
        sweep=CircularSweep(; max_sweeps=1),
        solver=CholeskySolve(),
    )
    B1 = split(A, algorithm).mps

    @test length(B1) == 4
    @test assertconsistency(B1) === nothing
    @test scalartype(B1[1]) == ComplexF64
    @test dense(B1) ≈ dense(A) rtol=1e-6
end

@testset "split — exact B0 stays unchanged after full sweep" begin
    B0 = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    A = contract_pairs(B0)
    ctx = test_split_state(A, deepcopy(B0))
    for st in LatticeIteratorLab.sweep_states(CircularSweep(; max_sweeps=1), 4)
        update!(ctx, st)
    end
    @test squared_distance(ctx.approximation, B0) ≈ 0 atol=1e-10
end

@testset "split — RandomInitialization + Z2 spaces, zigzag sweep" begin
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA = rand(PeriodicMPS, ComplexF64, 2, P ⊗ P, V)
    algorithm = ALSSplit(;
        initialization=RandomInitialization(; virtual_spaces=fill(V, 4)),
        sweep=ZigzagSweep(; max_sweeps=1),
        solver=LUSolve(),
    )
    B1 = split(mpsA, algorithm).mps

    @test length(B1) == 4
    @test assertconsistency(B1) === nothing
    @test scalartype(B1[1]) == ComplexF64
end

@testset "update! — exact periodic cost descends for both sweep orders" begin
    Random.seed!(1234)
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    initialization = SVDInitialization(; trunc=TruncationStrategy(maxrank=1))

    for sweep in (
            CircularSweep(; max_sweeps=2),
            ZigzagSweep(; max_sweeps=2),
        )
        ctx = test_split_state(mpsA, initial_approximation(mpsA, initialization); solver=LUSolve())
        previous_cost = squared_distance(mpsA, ctx.approximation)

        for st in LatticeIteratorLab.sweep_states(sweep, length(ctx.approximation))
            _, local_cost = update!(ctx, st)
            exact_cost = squared_distance(mpsA, ctx.approximation)
            @test local_cost ≈ exact_cost rtol=1e-9 atol=1e-9
            @test exact_cost <= previous_cost + 1e-9 * max(previous_cost, 1.0)
            previous_cost = exact_cost
        end
    end
end

@testset "split — sweep progress log" begin
    Random.seed!(4321)
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    nsweeps = 3
    algorithm = ALSSplit(;
        initialization=SVDInitialization(; trunc=TruncationStrategy(maxrank=1)),
        sweep=CircularSweep(; max_sweeps=nsweeps),
        solver=LUSolve(),
    )
    result, updates = collect_split(mpsA, algorithm)
    mpsB = result.mps
    sweep_costs = getfield.(updates[4:4:end], :local_cost)

    @test length(updates) == 4nsweeps
    @test length(sweep_costs) == nsweeps
    @test all(diff(sweep_costs) .<= 1e-9 .* max.(sweep_costs[1:(end - 1)], 1.0))
    @test sweep_costs[end] ≈ squared_distance(mpsA, mpsB) rtol=1e-10 atol=1e-10
    @test all(event.duration >= 0 for event in updates)
end
