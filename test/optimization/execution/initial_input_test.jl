using PMPSLab: SVDInitialization, SplitEnvironmentPolicy, CostTolerance
using LinearAlgebra: norm
using Test, Random, TensorKit, PMPSLab
using PMPSLab: initialize_split, restore_split
using PMPSLab: squared_distance, TruncatedTransferPolicy

@testset "split infers its domain and owns a copy of the initial tensors" begin
    Random.seed!(461)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    initial = initial_approximation(target, SVDInitialization(; trunc=truncrank(1)))
    original = deepcopy(initial)
    algorithm = ALSSplit(; sweep=CircularSweep(; max_sweeps=1), solver=LUSolve())
    run = initialize_split(target, initial, algorithm)
    @test length(run.sweep) == 4
    @test run.approximation !== initial
    @test all(run.approximation[i] !== initial[i] for i in eachindex(initial))
    fetch(start_run(PMPSLab.split_steps(run); threaded=false,
        onstep=(_, event) -> begin
            if event.completed_updates == 1
                @test dense(initial) == dense(original)
                @test squared_distance(target, run.approximation) <= squared_distance(target, initial) + 1e-10
            end
        end))
    result = split(target, initial, algorithm)
    inplace = deepcopy(initial)
    borrowed = initialize_split(target, inplace, algorithm; copy_initial=false)
    @test borrowed.approximation === inplace
    first_tensor = inplace[1]
    borrowed_result = fetch(start_run(PMPSLab.split_steps(borrowed); threaded=false,
        onstep=(_, event) -> begin
            event.completed_updates == 1 && @test inplace[1] !== first_tensor
        end))
    @test borrowed_result.mps === inplace
    @test dense(inplace) ≈ dense(result.mps)
    direct = deepcopy(initial)
    direct_result = split(target, direct, algorithm; copy_initial=false)
    @test direct_result.mps === direct
    @test dense(direct_result.mps) ≈ dense(result.mps)
    @test result.progress.completed_updates == 4
    @test dense(initial) == dense(original)
    @test all(domain(result.mps[i]) == domain(initial[i]) for i in eachindex(initial))
    @test_throws DimensionMismatch initialize_split(target, rand(PeriodicMPS, ComplexF64, 2, 2, 1), algorithm)
    @test_throws DimensionMismatch initialize_split(target, rand(PeriodicMPS, ComplexF64, 4, 3, 1), algorithm)
    other_target = rand(PeriodicMPS, ComplexF64, 3, (2, 2), 1)
    @test length(initialize_split(other_target, algorithm).sweep) == 6
end

@testset "exact errors are distinct from compressed local estimates" begin
    Random.seed!(462)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    initial = rand(PeriodicMPS, ComplexF64, 4, 2, 1)
    policy = SplitEnvironmentPolicy(; mixed=TruncatedTransferPolicy(; trunc=truncrank(1)))
    for measure in (:local, :exact)
        algorithm = ALSSplit(; sweep=CircularSweep(; max_sweeps=2), solver=LUSolve(),
            environments=policy, error_measure=measure)
        run = initialize_split(target, initial, algorithm)
        snapshots = [ExperimentLab.copy_state(run)]
        function check_error(active, record)
            @test record === active.progress
            if !record.sweep_completed
                @test isnothing(record.exact_error)
            end
            if record.completed_updates < 4
                @test active.progress.completed_sweeps == 0
            elseif record.completed_updates == 4
                @test record.sweep == 1
                @test record.local_cost == active.progress.local_cost
                if measure == :exact
                    @test record.exact_error ≈ norm(dense(target) - dense(active.approximation))^2
                    @test !isapprox(record.exact_error, record.local_cost; rtol=1e-5)
                else
                    @test isnothing(record.exact_error)
                end
            end
        end
        expected = fetch(start_run(PMPSLab.split_steps(run); threaded=false,
            onstep=(_, event) -> begin
                check_error(run, event)
                event.completed_updates == 3 && push!(snapshots, ExperimentLab.copy_state(run))
            end))
        restored = restore_split(snapshots[end], target)
        actual = fetch(start_run(PMPSLab.split_steps(restored); threaded=false,
            onstep=(_, event) -> check_error(restored, event)))
        @test dense(expected.mps) ≈ dense(actual.mps)
        @test run.progress.completed_sweeps == 2
    end
    @test_throws ArgumentError ALSSplit(; sweep=CircularSweep(; max_sweeps=1), solver=LUSolve(), error_measure=:unknown)
end

@testset "stopping uses the configured error measurement" begin
    Random.seed!(463)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    initial = rand(PeriodicMPS, ComplexF64, 4, 2, 1)
    environments = SplitEnvironmentPolicy(mixed=TruncatedTransferPolicy(trunc=truncrank(1)))
    probe = ALSSplit(sweep=CircularSweep(max_sweeps=2), solver=LUSolve(),
        environments=environments, error_measure=:exact)
    probe_result = split(target, initial, probe)
    approximation = probe_result.mps
    local_error = abs(probe_result.progress.local_cost)
    exact_error = norm(dense(target) - dense(approximation))^2
    @test !isapprox(local_error, exact_error; rtol=1e-5)
    threshold = (local_error + exact_error) / 2
    for measure in (:local, :exact)
        algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=2), solver=LUSolve(),
            environments=environments, error_measure=measure,
            stopping=CostTolerance(atol=threshold, rtol=1e12, min_sweeps=2))
        run = initialize_split(target, initial, algorithm)
        fetch(start_run(PMPSLab.split_steps(run); threaded=false))
        expected_error = measure === :exact ? exact_error : local_error
        @test run.progress.converged == (expected_error < threshold)
        restored = restore_split(ExperimentLab.copy_state(run), target)
        events = PMPSLab.SplitProgress[]
        fetch(start_run(PMPSLab.split_steps(restored); threaded=false,
            onstep=(_, event) -> push!(events, event)))
        @test isempty(events)
        @test restored.progress.converged == run.progress.converged
    end
end
