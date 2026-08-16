@enum LoopTNRPhase::UInt8 begin
    TNR_INITIALIZE
    TNR_PREPARE_SPLIT
    TNR_OPTIMIZE
    TNR_COARSE_GRAIN
    TNR_FINISHED
end

mutable struct LoopTNRRun{
        A<:LoopTNR,
        G<:GenericGateTensor,
        S<:ALSSplit,
        H<:MeasurementHistory,
    } <: AbstractAlgorithmRun
    algorithm::A
    cell::PlaquetteGates{G}
    completed_rg::Int
    phase::LoopTNRPhase
    active_split::Union{Nothing,SplitRun{S}}
    log::LoopTNRLog
    measurement_results::H
end

struct LoopTNRResult{
        G<:GenericGateTensor,
        H<:MeasurementHistory,
    }
    cell::PlaquetteGates{G}
    measurement_results::H
    log::LoopTNRLog
end

struct LoopTNRCoarseGrained{L<:RGLog,R<:TNRMeasurementResult} <: AlgorithmEvent
    rg::Int
    log::L
    result::R
end

struct LoopTNRSplitStepCompleted{E<:SplitSiteCompleted} <: AlgorithmEvent
    rg::Int
    event::E
end

const LoopTNRInputCell = Union{
    UniformGates,
    CheckerboardGates,
    PlaquetteGates,
}

function _looptnr_cell(cell::LoopTNRInputCell)
    return PlaquetteGates(cell)
end

function initialize_run(cell::LoopTNRInputCell, algorithm::LoopTNR)
    internal_cell = _looptnr_cell(cell)
    measurement_results = MeasurementHistory{TNRMeasurementResult}()
    return LoopTNRRun{
        typeof(algorithm),
        eltype(internal_cell),
        typeof(algorithm.split),
        typeof(measurement_results),
    }(
        algorithm,
        internal_cell,
        -1,
        TNR_INITIALIZE,
        nothing,
        LoopTNRLog(),
        measurement_results,
    )
end

isfinished(run::LoopTNRRun) = run.phase == TNR_FINISHED

function _initialize_tnr_step(run::LoopTNRRun)
    candidate = copy(run.cell)

    normalization_result = normalize!(
        candidate, run.algorithm.measurement.normalization
    )

    fixed_point_result = _fixed_point_measurement(
        run.algorithm.measurement.fixed_point, candidate
    )
    measurement_result = TNRMeasurementResult(
        0, normalization_result, fixed_point_result
    )
    rg_log = RGLog(0, nothing, 0.0)

    run.cell = candidate
    push!(run.log.rg_logs, rg_log)
    push!(run.measurement_results, measurement_result)
    run.completed_rg = 0
    run.phase = run.algorithm.max_rg_steps == 0 ?
        TNR_FINISHED : TNR_PREPARE_SPLIT
    return LoopTNRCoarseGrained(0, rg_log, measurement_result)
end

function _step_active_split!(run::LoopTNRRun)
    active_split = something(run.active_split)
    return _step_active_split!(run, active_split)
end

function _step_active_split!(run::LoopTNRRun, active_split::R) where {R<:SplitRun}
    rg = run.completed_rg + 1
    event = step!(active_split)
    isfinished(active_split) && (run.phase = TNR_COARSE_GRAIN)
    return LoopTNRSplitStepCompleted(rg, event)
end

function _prepare_split_step!(run::LoopTNRRun)
    target = PeriodicMPS(run.cell)
    active_split = initialize_run(target, run.algorithm.split)
    run.active_split = active_split

    if isfinished(active_split)
        run.phase = TNR_COARSE_GRAIN
        return _coarse_grain_tnr_step!(run)
    end

    run.phase = TNR_OPTIMIZE
    return _step_active_split!(run)
end

function _coarse_grain_tnr_step!(run::LoopTNRRun)
    active_split = something(run.active_split)
    return _coarse_grain_tnr_step!(run, active_split)
end

function _coarse_grain_tnr_step!(
        run::LoopTNRRun,
        active_split::R,
    ) where {R<:SplitRun}
    isfinished(active_split) || throw(ArgumentError(
        "LoopTNR cannot coarse grain an unfinished active split"
    ))

    split_result = run_result(active_split)
    started_at = time()
    coarse = _coarse_grain(split_result.mps)
    candidate = PlaquetteGates(coarse)
    coarse_grain_duration = time() - started_at
    assertconsistency(candidate)

    normalization_result = normalize!(
        candidate, run.algorithm.measurement.normalization
    )
    fixed_point_result = _fixed_point_measurement(
        run.algorithm.measurement.fixed_point, candidate
    )
    rg = run.completed_rg + 1
    measurement_result = TNRMeasurementResult(
        rg, normalization_result, fixed_point_result
    )
    rg_log = RGLog(rg, split_result.log, coarse_grain_duration)

    run.cell = candidate
    push!(run.log.rg_logs, rg_log)
    push!(run.measurement_results, measurement_result)
    run.completed_rg = rg
    run.active_split = nothing
    run.phase = rg == run.algorithm.max_rg_steps ?
        TNR_FINISHED : TNR_PREPARE_SPLIT
    return LoopTNRCoarseGrained(rg, rg_log, measurement_result)
end

function step!(run::LoopTNRRun)
    isfinished(run) && throw(ArgumentError("LoopTNR run is already complete"))
    run.phase == TNR_INITIALIZE && return _initialize_tnr_step(run)
    run.phase == TNR_PREPARE_SPLIT && return _prepare_split_step!(run)
    run.phase == TNR_OPTIMIZE && return _step_active_split!(run)
    run.phase == TNR_COARSE_GRAIN && return _coarse_grain_tnr_step!(run)
    throw(ArgumentError("invalid LoopTNR phase $(run.phase)"))
end

function run_result(run::LoopTNRRun)
    isfinished(run) || throw(ArgumentError("LoopTNR run is not complete"))
    return LoopTNRResult(run.cell, run.measurement_results, run.log)
end

"""
    renormalize(cell, algorithm::LoopTNR; observer=_DEFAULT_OBSERVER)

Run LoopTNR to completion through the generic resumable execution interface.
The caller's tensor cell is not modified.
"""
function renormalize(
        cell::LoopTNRInputCell,
        algorithm::LoopTNR;
        observer = _DEFAULT_OBSERVER,
    )
    run = initialize_run(cell, algorithm)
    resolved_observer = _resolve_observer(run, observer)
    try
        _observer_running!(resolved_observer, run)
        while !isfinished(run)
            event = step!(run)
            _observer_step!(resolved_observer, event, run)
        end
        _observer_completed!(resolved_observer, run)
    catch err
        _observer_failed!(resolved_observer, run, err)
        rethrow()
    end
    return run_result(run)
end
