struct GateCountMeasurement <: AbstractFixedPointMeasurement end

struct GateCountResult <: AbstractMeasurementResult
    count::Int
end

PMPSLab.measure(::GateCountMeasurement, cell::AbstractTensorCell) =
    GateCountResult(length(cell))

struct FailingFixedPointMeasurement <: AbstractFixedPointMeasurement end
PMPSLab.measure(::FailingFixedPointMeasurement, ::AbstractTensorCell) =
    error("injected fixed-point failure")

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
    algorithm = LoopTNR(rg0_split(); maximal_rg=0)

    for cell in rg0_cell_variants()
        input_before = copy.(collect(cell))
        input_identities = objectid.(collect(cell))
        run = initialize_run(cell, algorithm)

        @test run.cell !== cell
        @test run.completed_rg == -1
        @test run.phase == PMPSLab.TNR_INITIALIZE
        @test isnothing(run.active_split)
        @test !isfinished(run)
        @test_throws ArgumentError run_result(run)

        event = step!(run)
        result = run_result(run)

        @test event isa LoopTNRCoarseGrained
        @test result isa LoopTNRResult
        @test result.cell isa AbstractTensorCell
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
    run = initialize_run(cell, LoopTNR(rg0_split(); maximal_rg=0))
    session = start_run(run; observer=nothing, threaded=false)
    result = fetch(session)

    @test run_status(session) == PMPSLab.LOCAL_COMPLETED
    @test length(result.measurement_results) == 1
    @test length(result.log.rg_logs) == 1
end

@testset "LoopTNR RG 0 — fixed-point result and observer event" begin
    cell = UniformGates(normalization_gate())
    measurement = TNRMeasurement(
        InfinityNormNormalization(), GateCountMeasurement()
    )
    algorithm = LoopTNR(rg0_split(); maximal_rg=0, measurement)
    events = AlgorithmEvent[]

    result = renormalize(
        cell, algorithm; observer=event -> push!(events, event)
    )

    @test length(events) == 1
    @test only(events) isa LoopTNRCoarseGrained
    @test only(result.measurement_results).fixed_point == GateCountResult(1)
end

@testset "LoopTNR RG 0 — failures do not commit" begin
    cell = UniformGates(set_gate_norm!(normalization_gate(), 4.0))
    measurement = TNRMeasurement(
        InfinityNormNormalization(), FailingFixedPointMeasurement()
    )
    algorithm = LoopTNR(rg0_split(); maximal_rg=0, measurement)
    run = initialize_run(cell, algorithm)
    committed_before = copy.(collect(run.cell))

    @test_throws ErrorException step!(run)
    @test run.completed_rg == -1
    @test run.phase == PMPSLab.TNR_INITIALIZE
    @test isempty(run.log.rg_logs)
    @test isempty(run.measurement_results)
    @test all(collect(run.cell) .== committed_before)
    @test norm(run.cell.gate, Inf) ≈ 4
end

@testset "LoopTNR RG 0 — positive RG count is not advertised" begin
    cell = UniformGates(normalization_gate())
    algorithm = LoopTNR(rg0_split(); maximal_rg=1)
    @test_throws ArgumentError initialize_run(cell, algorithm)
    @test_throws ArgumentError renormalize(cell, algorithm; observer=nothing)
end
