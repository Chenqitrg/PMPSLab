struct LoopTNRProblemIdentity{A<:LoopTNR}
    algorithm::A
end

"""
    LoopTNRSnapshot

Durable safe-boundary state of a [`LoopTNRRun`](@ref). When a ALS split is
active, `active_split` contains its nested [`SplitSnapshot`](@ref); committed RG
measurements remain in the outer `measurement_results` history.
"""
struct LoopTNRSnapshot{
        G<:GenericGateTensor,
        P<:Union{Nothing,SplitSnapshot},
        H<:MeasurementHistory,
        I<:LoopTNRProblemIdentity,
    } <: AbstractAlgorithmSnapshot
    cell::PlaquetteGates{G}
    completed_rg::Int
    phase::LoopTNRPhase
    active_split::P
    log::LoopTNRLog
    measurement_results::H
    identity::I
end

_split_state_finished(run::SplitRun) = isfinished(run)
_split_state_finished(saved::SplitSnapshot) =
    isnothing(saved.pending_iteration)

function _validate_looptnr_records(
        completed_rg::Int,
        log::LoopTNRLog,
        measurement_results::MeasurementHistory,
    )
    expected_length = completed_rg + 1
    length(log.rg_logs) == expected_length || throw(ArgumentError(
        "LoopTNR state has $(length(log.rg_logs)) RG logs; expected $expected_length"
    ))
    length(measurement_results) == expected_length || throw(ArgumentError(
        "LoopTNR state has $(length(measurement_results)) measurement results; " *
        "expected $expected_length"
    ))

    for (index, rg_log) in enumerate(log.rg_logs)
        expected_rg = index - 1
        rg_log.rg == expected_rg || throw(ArgumentError(
            "LoopTNR log index $index records RG $(rg_log.rg), expected $expected_rg"
        ))
        if expected_rg == 0
            isnothing(rg_log.split_log) || throw(ArgumentError(
                "LoopTNR RG 0 log must not contain a split log"
            ))
        else
            rg_log.split_log isa SplitLog || throw(ArgumentError(
                "LoopTNR RG $expected_rg log must contain a SplitLog"
            ))
        end
    end

    for (index, result) in enumerate(measurement_results)
        expected_rg = index - 1
        result isa TNRMeasurementResult || throw(ArgumentError(
            "LoopTNR measurement index $index has unsupported type $(typeof(result))"
        ))
        result.rg == expected_rg || throw(ArgumentError(
            "LoopTNR measurement index $index records RG $(result.rg), " *
            "expected $expected_rg"
        ))
    end
    return nothing
end

function _validate_looptnr_state(
        completed_rg::Int,
        phase::LoopTNRPhase,
        active_split,
        log::LoopTNRLog,
        measurement_results::MeasurementHistory,
        algorithm::LoopTNR,
    )
    if phase == TNR_INITIALIZE
        completed_rg == -1 || throw(ArgumentError(
            "LoopTNR initialize phase requires completed_rg=-1"
        ))
        isnothing(active_split) || throw(ArgumentError(
            "LoopTNR initialize phase cannot contain an active split"
        ))
        return _validate_looptnr_records(
            completed_rg, log, measurement_results
        )
    end

    0 <= completed_rg <= algorithm.max_rg_steps || throw(ArgumentError(
        "LoopTNR completed_rg=$completed_rg is outside 0:$(algorithm.max_rg_steps)"
    ))
    _validate_looptnr_records(completed_rg, log, measurement_results)

    if phase == TNR_PREPARE_SPLIT
        completed_rg < algorithm.max_rg_steps || throw(ArgumentError(
            "LoopTNR prepare-split phase requires another RG step"
        ))
        isnothing(active_split) || throw(ArgumentError(
            "LoopTNR prepare-split phase cannot contain an active split"
        ))
    elseif phase == TNR_OPTIMIZE
        completed_rg < algorithm.max_rg_steps || throw(ArgumentError(
            "LoopTNR optimize phase requires another RG step"
        ))
        isnothing(active_split) && throw(ArgumentError(
            "LoopTNR optimize phase requires an active split"
        ))
        _split_state_finished(active_split) && throw(ArgumentError(
            "LoopTNR optimize phase requires an unfinished active split"
        ))
    elseif phase == TNR_COARSE_GRAIN
        completed_rg < algorithm.max_rg_steps || throw(ArgumentError(
            "LoopTNR coarse-grain phase requires another RG step"
        ))
        isnothing(active_split) && throw(ArgumentError(
            "LoopTNR coarse-grain phase requires a completed active split"
        ))
        _split_state_finished(active_split) || throw(ArgumentError(
            "LoopTNR coarse-grain phase requires a completed active split"
        ))
    elseif phase == TNR_FINISHED
        completed_rg == algorithm.max_rg_steps || throw(ArgumentError(
            "LoopTNR finished phase requires completed_rg=$(algorithm.max_rg_steps)"
        ))
        isnothing(active_split) || throw(ArgumentError(
            "LoopTNR finished phase cannot contain an active split"
        ))
    else
        throw(ArgumentError("unsupported LoopTNR phase $phase"))
    end
    return nothing
end

function snapshot(run::LoopTNRRun)
    _validate_looptnr_state(
        run.completed_rg,
        run.phase,
        run.active_split,
        run.log,
        run.measurement_results,
        run.algorithm,
    )
    assertconsistency(run.cell)
    active_split = isnothing(run.active_split) ?
        nothing : snapshot(run.active_split)
    return LoopTNRSnapshot(
        copy(run.cell),
        run.completed_rg,
        run.phase,
        active_split,
        deepcopy(run.log),
        deepcopy(run.measurement_results),
        LoopTNRProblemIdentity(deepcopy(run.algorithm)),
    )
end

"""
    restore_run(saved::LoopTNRSnapshot, algorithm::LoopTNR) -> LoopTNRRun

Restore a LoopTNR root run at exactly the next safe step. A nested split is
rebuilt from `saved.active_split` and the target derived from the saved committed
tensor cell.
"""
function restore_run(saved::LoopTNRSnapshot, algorithm::LoopTNR)
    saved.identity.algorithm == algorithm || throw(ArgumentError(
        "LoopTNR snapshot does not match the supplied algorithm configuration"
    ))
    assertconsistency(saved.cell)
    _validate_looptnr_state(
        saved.completed_rg,
        saved.phase,
        saved.active_split,
        saved.log,
        saved.measurement_results,
        algorithm,
    )

    cell = copy(saved.cell)
    active_split = if isnothing(saved.active_split)
        nothing
    else
        target = PeriodicMPS(cell)
        restore_run(saved.active_split, target, algorithm.split)
    end
    measurement_results = deepcopy(saved.measurement_results)
    run = LoopTNRRun{
        typeof(algorithm),
        eltype(cell),
        typeof(algorithm.split),
        typeof(measurement_results),
    }(
        algorithm,
        cell,
        saved.completed_rg,
        saved.phase,
        active_split,
        deepcopy(saved.log),
        measurement_results,
    )
    _validate_looptnr_state(
        run.completed_rg,
        run.phase,
        run.active_split,
        run.log,
        run.measurement_results,
        run.algorithm,
    )
    return run
end
