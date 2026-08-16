@testset "algorithm configurations have value semantics" begin
    virtual_space = Z2Space(0 => 2, 1 => 1)
    configurations = (
        RandomAnsatz(; virtual_spaces=fill(virtual_space, 4)),
        OneAnsatz(; virtual_spaces=fill(virtual_space, 4)),
        SVDAnsatz(; trunc=TruncationStrategy()),
        PeriodicGaugeAnsatz(;
            initial_split=SVDAnsatz(; trunc=TruncationStrategy()),
            gauge_fix=ProjectorGaugeFix(),
        ),
        CircularSweep(; site_count=4, max_sweeps=2),
        ZigzagSweep(; site_count=4, max_sweeps=2),
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
        @test configuration isa PMPSLab.AbstractAlgorithmConfig
        @test configuration == copied
        @test isequal(configuration, copied)
        @test hash(configuration) == hash(copied)
    end

    algorithm = ALSSplit(;
        ansatz=first(configurations),
        sweep=CircularSweep(; site_count=4, max_sweeps=1),
        solver=LUSolve(),
    )
    @test algorithm == deepcopy(algorithm)
    @test hash(algorithm) == hash(deepcopy(algorithm))
end

@testset "algorithm configurations — recursive plain-text display" begin
    ansatz = PeriodicGaugeAnsatz()
    ansatz_text = sprint(show, MIME"text/plain"(), ansatz)

    @test startswith(ansatz_text, "PeriodicGaugeAnsatz:\n")
    @test occursin(
        "├─ initial_split: SVDAnsatz\n│  └─ trunc: ",
        ansatz_text,
    )
    @test occursin(
        "└─ gauge_fix: ProjectorGaugeFix\n" *
            "   ├─ tol: 1.0e-12\n" *
            "   ├─ maxiter: 100\n" *
            "   ├─ trunc: ",
        ansatz_text,
    )
    @test occursin("└─ trunc: notrunc()", ansatz_text)
    @test occursin("├─ trunc: notrunc()", ansatz_text)
    @test occursin("└─ orthogonalization_algorithm: Householder", ansatz_text)
    @test !occursin("DefaultDriver", ansatz_text)
    @test !occursin("initial_split: SVDAnsatz{", ansatz_text)
    @test sprint(show, ansatz) == "PeriodicGaugeAnsatz(...)"
    @test sprint(show, LUSolve()) == "LUSolve()"
    @test sprint(
        show,
        MIME"text/plain"(),
        ansatz;
        context = :compact => true,
    ) == "PeriodicGaugeAnsatz(...)"

    split_algorithm = ALSSplit(;
        ansatz,
        sweep=CircularSweep(; site_count=4, max_sweeps=1),
        solver=CholeskySolve(),
    )
    split_text = sprint(show, MIME"text/plain"(), split_algorithm)
    @test occursin(
        "├─ ansatz: PeriodicGaugeAnsatz\n" *
            "│  ├─ initial_split: SVDAnsatz\n" *
            "│  │  └─ trunc: ",
        split_text,
    )
    @test occursin("├─ solver: CholeskySolve()\n", split_text)
    @test occursin(
        "├─ environments: SplitEnvironmentPolicy\n" *
            "│  ├─ normal: DenseTransferPolicy()\n" *
            "│  └─ mixed: DenseTransferPolicy()\n" *
            "└─ stopping: NoStopping()\n",
        split_text,
    )

    looptnr_text = sprint(
        show,
        MIME"text/plain"(),
        default_looptnr(; max_rg_steps=4, dcut=16),
    )
    @test occursin(
        "trunc: truncrank(32) & truncerror(rtol=1.0e-13)",
        looptnr_text,
    )
    @test occursin(
        "trunc: truncrank(16) & truncerror(rtol=1.0e-13)",
        looptnr_text,
    )
    @test occursin("orthogonalization_algorithm: Householder", looptnr_text)
    @test !occursin("TruncationIntersection", looptnr_text)
    @test !occursin("DefaultDriver", looptnr_text)
    @test sprint(
        show,
        default_looptnr(; max_rg_steps=4, dcut=16),
    ) == "LoopTNR(...)"

    uncommon_truncation = SVDAnsatz(;
        trunc=truncrank(2) | truncerror(; rtol=1e-8),
    )
    @test occursin(
        "trunc: TruncationUnion(...)",
        sprint(show, MIME"text/plain"(), uncommon_truncation),
    )
end

@testset "split — SVDAnsatz, circular sweep, exact solve" begin
    B0 = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    A = contract_pairs(B0)
    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(; trunc=TruncationStrategy()),
        sweep=CircularSweep(; site_count=4, max_sweeps=1),
        solver=CholeskySolve(),
    )
    B1, log = split(A, algorithm)

    @test length(B1) == 4
    @test assertconsistency(B1) === nothing
    @test scalartype(B1[1]) == ComplexF64
    @test dense(B1) ≈ dense(A) rtol=1e-6
end

@testset "split — exact B0 stays unchanged after full sweep" begin
    B0 = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    A = contract_pairs(B0)
    ctx = SplitContext(A, deepcopy(B0))
    for st in CircularSweep(; site_count=4, max_sweeps=1)
        update!(ctx, CholeskySolve(), st)
    end
    @test squared_distance(ctx.approximation, B0) ≈ 0 atol=1e-10
end

@testset "split — RandomAnsatz + Z2 spaces, zigzag sweep" begin
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA = rand(PeriodicMPS, ComplexF64, 2, P ⊗ P, V)
    algorithm = ALSSplit(;
        ansatz=RandomAnsatz(; virtual_spaces=fill(V, 4)),
        sweep=ZigzagSweep(; site_count=4, max_sweeps=1),
        solver=LUSolve(),
    )
    B1, log = split(mpsA, algorithm)

    @test length(B1) == 4
    @test assertconsistency(B1) === nothing
    @test scalartype(B1[1]) == ComplexF64
end

@testset "update! — exact periodic cost descends for both sweep orders" begin
    Random.seed!(1234)
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    ansatz = SVDAnsatz(; trunc=TruncationStrategy(maxrank=1))

    for sweep in (
            CircularSweep(; site_count=4, max_sweeps=2),
            ZigzagSweep(; site_count=4, max_sweeps=2),
        )
        ctx = SplitContext(mpsA, initial_approximation(mpsA, ansatz))
        previous_cost = squared_distance(mpsA, ctx.approximation)

        for st in sweep
            _, local_cost = update!(ctx, LUSolve(), st)
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
        ansatz=SVDAnsatz(; trunc=TruncationStrategy(maxrank=1)),
        sweep=CircularSweep(; site_count=4, max_sweeps=nsweeps),
        solver=LUSolve(),
    )
    mpsB, log = split(mpsA, algorithm)
    sweep_costs = getfield.(log.site_logs[4:4:end], :cost)

    @test length(log.site_logs) == 4nsweeps
    @test length(sweep_costs) == nsweeps
    @test all(diff(sweep_costs) .<= 1e-9 .* max.(sweep_costs[1:(end - 1)], 1.0))
    @test sweep_costs[end] ≈ squared_distance(mpsA, mpsB) rtol=1e-10 atol=1e-10
    @test log.duration >= 0
end
