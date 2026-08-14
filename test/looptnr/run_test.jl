struct GateCountMeasurement <: AbstractFixedPointMeasurement end

struct GateCountResult <: AbstractMeasurementResult
    count::Int
end

PMPSLab.measure(::GateCountMeasurement, cell::AbstractTensorCell) =
    GateCountResult(length(cell))

struct FailingFixedPointMeasurement <: AbstractFixedPointMeasurement end
PMPSLab.measure(::FailingFixedPointMeasurement, ::AbstractTensorCell) =
    error("injected fixed-point failure")

mutable struct FailingCoarseMeasurement <: AbstractFixedPointMeasurement
    calls::Int
end
FailingCoarseMeasurement() = FailingCoarseMeasurement(0)

function PMPSLab.measure(
        measurement::FailingCoarseMeasurement,
        cell::PlaquetteGates,
    )
    measurement.calls += 1
    measurement.calls == 1 || error(
        "injected coarse-grained fixed-point failure"
    )
    return GateCountResult(length(cell))
end

function rg0_cell_variants()
    uniform_gate = normalization_gate()
    checkerboard_gate = normalization_gate()
    plaquette_gate = normalization_gate()
    return (
        UniformGates(uniform_gate),
        CheckerboardGates(checkerboard_gate, checkerboard_gate),
        PlaquetteGates(fill(plaquette_gate, 2, 2); copy=false),
    )
end

@testset "LoopTNR RG 0 — all tensor-cell variants" begin
    algorithm = LoopTNR(; split=rg0_split(), maximal_rg=0)

    for cell in rg0_cell_variants()
        input_before = copy.(collect(cell))
        input_identities = objectid.(collect(cell))
        run = initialize_run(cell, algorithm)

        @test run.cell !== cell
        @test run.cell isa PlaquetteGates
        @test fieldtype(typeof(run), :cell) == typeof(run.cell)
        @test all(
            stored !== source
            for stored in collect(run.cell), source in collect(cell)
        )
        @test length(unique(objectid, collect(run.cell))) == 4
        @test fieldtype(typeof(run), :active_split) ==
            Union{Nothing,SplitRun{typeof(algorithm.split)}}
        @test run.measurement_results isa
            MeasurementHistory{TNRMeasurementResult}
        @test eltype(run.log.rg_logs) ==
            Union{RGLog{Nothing},RGLog{SplitLog}}
        @test run.completed_rg == -1
        @test run.phase == PMPSLab.TNR_INITIALIZE
        @test isnothing(run.active_split)
        @test !isfinished(run)
        @test_throws ArgumentError run_result(run)

        event = step!(run)
        result = run_result(run)

        @test event isa LoopTNRCoarseGrained
        @test result isa LoopTNRResult
        @test result.cell isa PlaquetteGates
        @test result.measurement_results isa MeasurementHistory
        @test result.log isa LoopTNRLog
        @test event.rg == 0
        @test event.result === only(result.measurement_results)
        @test event.log === only(result.log.rg_logs)
        @test event.log.split_log === nothing
        @test event.log.coarse_grain_time == 0
        @test fieldnames(typeof(event.log)) == (
            :rg,
            :split_log,
            :coarse_grain_time,
        )
        @test isfinished(run)
        @test run.completed_rg == 0
        @test run.phase == PMPSLab.TNR_FINISHED
        @test length(result.measurement_results) == 1
        @test length(result.log.rg_logs) == 1
        @test exp(sum(log, independent_gate_norms(result.cell)) /
                  length(independent_gate_norms(result.cell))) ≈ 1
        @test objectid.(collect(cell)) == input_identities
        @test all(collect(cell) .== input_before)
        @test_throws ArgumentError step!(run)
    end
end

@testset "LoopTNR RG 0 — generic local execution" begin
    cell = UniformGates(normalization_gate())
    run = initialize_run(cell, LoopTNR(; split=rg0_split(), maximal_rg=0))
    session = start_run(run; observer=nothing, threaded=false)
    result = fetch(session)

    @test run_status(session) == PMPSLab.LOCAL_COMPLETED
    @test length(result.measurement_results) == 1
    @test length(result.log.rg_logs) == 1
end

@testset "LoopTNR RG 0 — fixed-point result and observer event" begin
    cell = UniformGates(normalization_gate())
    measurement = TNRMeasurement(;
        normalization=InfinityNormNormalization(),
        fixed_point=GateCountMeasurement(),
    )
    algorithm = LoopTNR(; split=rg0_split(), maximal_rg=0, measurement)
    events = AlgorithmEvent[]

    result = renormalize(
        cell, algorithm; observer=event -> push!(events, event)
    )

    @test length(events) == 1
    @test only(events) isa LoopTNRCoarseGrained
    @test only(result.measurement_results).fixed_point == GateCountResult(4)
end

@testset "LoopTNR RG 0 — failures do not commit" begin
    cell = UniformGates(set_gate_norm!(normalization_gate(), 4.0))
    measurement = TNRMeasurement(;
        normalization=InfinityNormNormalization(),
        fixed_point=FailingFixedPointMeasurement(),
    )
    algorithm = LoopTNR(; split=rg0_split(), maximal_rg=0, measurement)
    run = initialize_run(cell, algorithm)
    committed_before = copy.(collect(run.cell))

    @test_throws ErrorException step!(run)
    @test run.completed_rg == -1
    @test run.phase == PMPSLab.TNR_INITIALIZE
    @test isempty(run.log.rg_logs)
    @test isempty(run.measurement_results)
    @test all(collect(run.cell) .== committed_before)
    @test all(norm(gate, Inf) ≈ 4 for gate in run.cell)
end

@testset "LoopTNR multi-RG — zero-length split" begin
    cell = CheckerboardGates(normalization_gate(), normalization_gate())
    input_before = copy.(collect(cell))
    algorithm = LoopTNR(; split=rg0_split(), maximal_rg=2)
    events = AlgorithmEvent[]

    result = renormalize(
        cell, algorithm; observer=event -> push!(events, event)
    )

    @test length(events) == 3
    @test all(event -> event isa LoopTNRCoarseGrained, events)
    @test getfield.(events, :rg) == [0, 1, 2]
    @test length(result.measurement_results) == 3
    @test getfield.(result.measurement_results, :rg) == [0, 1, 2]
    @test length(result.log.rg_logs) == 3
    @test isnothing(result.log.rg_logs[1].split_log)
    @test all(isempty(log.split_log.site_logs) for log in result.log.rg_logs[2:3])
    @test result.cell isa PlaquetteGates
    @test isnothing(assertconsistency(result.cell))
    @test all(collect(cell) .== input_before)
end

@testset "LoopTNR RG 1 — all input tensor-cell variants" begin
    algorithm = LoopTNR(; split=rg0_split(), maximal_rg=1)

    for cell in rg0_cell_variants()
        result = renormalize(cell, algorithm; observer=nothing)
        @test result.cell isa PlaquetteGates
        @test isnothing(assertconsistency(result.cell))
        @test length(result.measurement_results) == 2
        @test length(result.log.rg_logs) == 2
    end
end

@testset "LoopTNR multi-RG — nested split safe boundaries" begin
    gate = ones(ComplexF64, ℂ^1 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1)
    cell = UniformGates(gate)
    split_algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; N=8, maximal_sweep=1),
        solve=CholeskySolve(),
    )
    run = initialize_run(cell, LoopTNR(; split=split_algorithm, maximal_rg=1))

    rg0_event = step!(run)
    @test rg0_event isa LoopTNRCoarseGrained
    @test run.phase == PMPSLab.TNR_PREPARE_SPLIT
    committed_cell = run.cell

    split_events = [step!(run) for _ in 1:8]
    @test all(event -> event isa LoopTNRSplitStepCompleted, split_events)
    @test getfield.(split_events, :rg) == fill(1, 8)
    @test getfield.([event.event.log for event in split_events], :state) ==
          [SweepState{Forward}(site, 0) for site in 1:8]
    @test run.cell === committed_cell
    @test run.phase == PMPSLab.TNR_COARSE_GRAIN
    @test !isnothing(run.active_split)
    @test isfinished(something(run.active_split))
    @test run.completed_rg == 0

    rg1_event = step!(run)
    @test rg1_event isa LoopTNRCoarseGrained
    @test rg1_event.rg == 1
    @test isfinished(run)
    @test isnothing(run.active_split)
    @test run.completed_rg == 1
    @test typeof(run.cell) == typeof(committed_cell)
    @test length(run.log.rg_logs[2].split_log.site_logs) == 8
    @test length(run.measurement_results) == 2
end

@testset "LoopTNR multi-RG — failed commit retains completed split" begin
    cell = UniformGates(normalization_gate())
    measurement = TNRMeasurement(;
        normalization=InfinityNormNormalization(),
        fixed_point=FailingCoarseMeasurement(),
    )
    run = initialize_run(
        cell,
        LoopTNR(; split=rg0_split(), maximal_rg=1, measurement),
    )
    step!(run)
    committed_cell = run.cell
    committed_log = only(run.log.rg_logs)
    committed_measurement = only(run.measurement_results)

    @test_throws ErrorException step!(run)
    @test run.phase == PMPSLab.TNR_COARSE_GRAIN
    @test run.completed_rg == 0
    @test run.cell === committed_cell
    @test only(run.log.rg_logs) === committed_log
    @test only(run.measurement_results) === committed_measurement
    @test !isnothing(run.active_split)
    @test isfinished(something(run.active_split))
end

@testset "LoopTNR multi-RG — local pause inside active split" begin
    gate = ones(ComplexF64, ℂ^1 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1)
    split_algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; N=8, maximal_sweep=1),
        solve=CholeskySolve(),
    )
    run = initialize_run(
        UniformGates(gate),
        LoopTNR(; split=split_algorithm, maximal_rg=1),
    )
    session_ref = Ref{LocalRunSession}()
    events = AlgorithmEvent[]
    observer = event -> begin
        push!(events, event)
        if event isa LoopTNRSplitStepCompleted &&
                event.event.log.state.site == 4
            pause!(session_ref[])
        end
        return nothing
    end
    session = LocalRunSession(run; observer, threaded=false)
    session_ref[] = session

    start_run!(session)
    wait(session)
    @test run_status(session) == PMPSLab.LOCAL_PAUSED
    @test run.phase == PMPSLab.TNR_OPTIMIZE
    @test length(something(run.active_split).log.site_logs) == 4

    resume!(session)
    result = fetch(session)
    @test run_status(session) == PMPSLab.LOCAL_COMPLETED
    @test length(events) == 10
    @test count(event -> event isa LoopTNRSplitStepCompleted, events) == 8
    @test getfield.(filter(event -> event isa LoopTNRCoarseGrained, events), :rg) ==
          [0, 1]
    @test length(result.measurement_results) == 2
end
