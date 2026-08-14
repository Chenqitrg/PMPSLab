@testset "algorithm configurations have value semantics" begin
    virtual_space = Z2Space(0 => 2, 1 => 1)
    configurations = (
        RandomAnsatz(virtual_space, 4),
        OneAnsatz(virtual_space, 4),
        SVDAnsatz(),
        PeriodicGaugeAnsatz(),
        CircularSweep(4; maximal_sweep=2),
        ZigzagSweep(4; maximal_sweep=2),
        LUSolve(),
        CholeskySolve(),
        KrylovSolve(GMRES(; maxiter=10)),
        CostConvergence(),
        SplitEnvironmentPolicy(),
        SplitEnvironmentPolicy(
            TruncatedTransferPolicy(TruncationStrategy())
        ),
    )

    for configuration in configurations
        copied = deepcopy(configuration)
        @test configuration isa PMPSLab.AbstractAlgorithmConfig
        @test configuration == copied
        @test isequal(configuration, copied)
        @test hash(configuration) == hash(copied)
    end

    algorithm = PeriodicSplit(
        first(configurations),
        CircularSweep(4; maximal_sweep=1),
        LUSolve(),
    )
    @test algorithm == deepcopy(algorithm)
    @test hash(algorithm) == hash(deepcopy(algorithm))
end

@testset "algorithm configurations — recursive plain-text display" begin
    ansatz = PeriodicGaugeAnsatz()
    ansatz_text = sprint(show, MIME"text/plain"(), ansatz)

    @test startswith(ansatz_text, "PeriodicGaugeAnsatz:\n")
    @test occursin(
        "├─ svd_pre: SVDAnsatz\n│  └─ truncation: ",
        ansatz_text,
    )
    @test occursin(
        "└─ canonical: ProjectorCanonical\n" *
            "   ├─ tol: 1.0e-12\n" *
            "   ├─ maxiter: 100\n" *
            "   ├─ trscheme: ",
        ansatz_text,
    )
    @test !occursin("svd_pre: SVDAnsatz{", ansatz_text)
    @test sprint(
        show,
        MIME"text/plain"(),
        ansatz;
        context = :compact => true,
    ) == "PeriodicGaugeAnsatz(...)"

    split_algorithm = PeriodicSplit(
        ansatz,
        CircularSweep(4; maximal_sweep=1),
        CholeskySolve(),
    )
    split_text = sprint(show, MIME"text/plain"(), split_algorithm)
    @test occursin(
        "├─ ansatz: PeriodicGaugeAnsatz\n" *
            "│  ├─ svd_pre: SVDAnsatz\n" *
            "│  │  └─ truncation: ",
        split_text,
    )
    @test occursin("├─ solve: CholeskySolve()\n", split_text)
    @test occursin(
        "├─ environments: SplitEnvironmentPolicy\n" *
            "│  ├─ norm: DenseTransferPolicy()\n" *
            "│  └─ mixed: DenseTransferPolicy()\n" *
            "└─ stopping: nothing\n",
        split_text,
    )
end

@testset "split — SVDAnsatz, circular sweep, exact solve" begin
    B0 = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
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
    @test dense(B1) ≈ dense(A) rtol=1e-6
end

@testset "split — exact B0 stays unchanged after full sweep" begin
    B0 = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
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
    mpsA = rand(PeriodicMPS, ComplexF64, 2, P ⊗ P, V)
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
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
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
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
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
