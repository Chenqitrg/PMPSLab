@testset "SplitSnapshot — captures an isolated safe-boundary state" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; N=4, maximal_sweep=1),
        solve=CholeskySolve(),
    )
    run = initialize_run(mpsA, algorithm)
    step!(run)
    push!(run.measurement_results, NormalizationResult(2.0, log(2.0)))

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
    @test saved.measurement_results !== run.measurement_results
    @test collect(saved.measurement_results) == collect(run.measurement_results)

    saved_next_tensor = copy(saved.mpsB[2])
    step!(run)
    @test saved.pending_iteration != run.pending_iteration
    @test length(saved.log.site_logs) == 1
    @test length(run.log.site_logs) == 2
    @test length(saved.measurement_results) == 1
    @test saved.mpsB[2] == saved_next_tensor

    while !isfinished(run)
        step!(run)
    end
    completed = snapshot(run)
    @test isnothing(completed.pending_iteration)
    @test length(completed.log.site_logs) == 4
end

@testset "SplitSnapshot — restores one mid-sweep boundary" begin
    mpsA = pair_contraction(rand(PeriodicMPS, ComplexF64, 4, 2, 2))
    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; N=4, maximal_sweep=1),
        solve=CholeskySolve(),
    )
    uninterrupted = initialize_run(mpsA, algorithm)
    step!(uninterrupted)
    step!(uninterrupted)

    saved = snapshot(uninterrupted)
    saved_tensor = copy(saved.mpsB[3])
    restored = restore_run(saved, mpsA, algorithm)

    @test restored !== uninterrupted
    @test restored.context.mpsB !== saved.mpsB
    @test all(restored.context.mpsB[i] !== saved.mpsB[i]
              for i in eachindex(saved.mpsB))
    @test restored.log !== saved.log
    @test restored.log.site_logs !== saved.log.site_logs
    @test restored.log.site_logs == saved.log.site_logs
    @test restored.pending_iteration == saved.pending_iteration

    while !isfinished(uninterrupted)
        step!(uninterrupted)
    end
    while !isfinished(restored)
        step!(restored)
    end

    expected_result = run_result(uninterrupted)
    restored_result = run_result(restored)
    expected_mps = expected_result.mps
    expected_log = expected_result.log
    restored_mps = restored_result.mps
    restored_log = restored_result.log
    @test dense(restored_mps) ≈ dense(expected_mps) rtol=1e-12
    @test getfield.(restored_log.site_logs, :error) ≈
          getfield.(expected_log.site_logs, :error)
    @test saved.mpsB[3] == saved_tensor
    @test length(saved.log.site_logs) == 2
end

@testset "SplitSnapshot — restores every circular and zigzag boundary" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    finish_run! = function(run)
        while !isfinished(run)
            step!(run)
        end
        return run_result(run)
    end

    for sweep in (
            CircularSweep(; N=4, maximal_sweep=2),
            ZigzagSweep(; N=4, maximal_sweep=2),
        )
        algorithm = ALSSplit(;
            ansatz=SVDAnsatz(),
            sweep,
            solve=CholeskySolve(),
        )
        for completed_steps in 0:length(sweep)
            uninterrupted = initialize_run(mpsA, algorithm)
            for _ in 1:completed_steps
                step!(uninterrupted)
            end

            restored = restore_run(snapshot(uninterrupted), mpsA, algorithm)
            expected_result = finish_run!(uninterrupted)
            restored_result = finish_run!(restored)
            expected_mps = expected_result.mps
            expected_log = expected_result.log
            restored_mps = restored_result.mps
            restored_log = restored_result.log

            @test dense(restored_mps) ≈ dense(expected_mps) rtol=1e-12
            @test getfield.(restored_log.site_logs, :error) ≈
                  getfield.(expected_log.site_logs, :error)
            @test length(restored_log.site_logs) == length(sweep)
        end
    end
end

@testset "SplitSnapshot — rejects an inconsistent cursor or sweep" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; N=4, maximal_sweep=2),
        solve=CholeskySolve(),
    )
    run = initialize_run(mpsA, algorithm)
    for _ in 1:3
        step!(run)
    end
    saved = snapshot(run)

    bad_cursor = SplitSnapshot(
        saved.mpsB,
        iterate(algorithm.sweep),
        saved.log,
        saved.measurement_results,
        saved.identity,
    )
    @test_throws ArgumentError restore_run(bad_cursor, mpsA, algorithm)

    wrong_sweep = ALSSplit(;
        ansatz=algorithm.ansatz,
        sweep=ZigzagSweep(; N=4, maximal_sweep=2),
        solve=algorithm.solve,
    )
    @test_throws ArgumentError restore_run(saved, mpsA, wrong_sweep)

    wrong_domain = ALSSplit(;
        ansatz=algorithm.ansatz,
        sweep=CircularSweep(; N=1, maximal_sweep=2),
        solve=algorithm.solve,
    )
    @test_throws DimensionMismatch restore_run(saved, mpsA, wrong_domain)
end

@testset "SplitSnapshot — restore requires compatible inputs" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; N=4, maximal_sweep=1),
        solve=CholeskySolve(),
    )
    saved = snapshot(initialize_run(mpsA, algorithm))

    different_target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    @test_nowarn restore_run(saved, different_target, algorithm)

    incompatible_target = rand(PeriodicMPS, ComplexF64, 2, (3, 3), 1)
    @test_throws ArgumentError restore_run(saved, incompatible_target, algorithm)

    different_solver = ALSSplit(;
        ansatz=algorithm.ansatz,
        sweep=algorithm.sweep,
        solve=LUSolve(),
    )
    @test_throws ArgumentError restore_run(saved, mpsA, different_solver)

    different_ansatz = ALSSplit(;
        ansatz=OneAnsatz(; virtual_space=fill(Z2Space(0 => 1), 4)),
        sweep=algorithm.sweep,
        solve=algorithm.solve,
    )
    @test_throws ArgumentError restore_run(saved, mpsA, different_ansatz)

    truncated = TruncatedTransferPolicy(; truncation=TruncationStrategy())
    different_environments = ALSSplit(;
        ansatz=algorithm.ansatz,
        sweep=algorithm.sweep,
        solve=algorithm.solve,
        environments=SplitEnvironmentPolicy(;
            norm=truncated,
            mixed=truncated,
        ),
    )
    @test_throws ArgumentError restore_run(saved, mpsA, different_environments)

    different_stopping = ALSSplit(;
        ansatz=algorithm.ansatz,
        sweep=algorithm.sweep,
        solve=algorithm.solve,
        stopping=CostConvergence(),
    )
    @test_throws ArgumentError restore_run(saved, mpsA, different_stopping)

    virtual_space = Z2Space(0 => 2, 1 => 1)
    physical_space = Z2Space(0 => 2)
    graded_target = rand(PeriodicMPS, ComplexF64, 2, physical_space ⊗ physical_space, virtual_space)
    random_algorithm = ALSSplit(;
        ansatz=RandomAnsatz(; virtual_space=fill(virtual_space, 4)),
        sweep=CircularSweep(; N=4, maximal_sweep=1),
        solve=LUSolve(),
    )
    random_saved = snapshot(initialize_run(graded_target, random_algorithm))
    @test_nowarn restore_run(
        random_saved,
        graded_target,
        deepcopy(random_algorithm),
    )
end
