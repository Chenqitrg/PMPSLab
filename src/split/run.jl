"""
    SplitRun

Mutable execution state for a [`ALSSplit`](@ref). `context` owns the
numerical working state, while the run owns the algorithm, sweep cursor,
and progress log.
"""
const SplitPendingIteration = Union{
    Nothing,
    Tuple{SweepState,Int},
}

mutable struct SplitRun{
        A<:ALSSplit,
        C<:SplitContext,
        P<:SplitPendingIteration,
        H<:MeasurementHistory,
    } <: AbstractAlgorithmRun
    algorithm::A
    context::C
    pending_iteration::P
    log::SplitLog
    measurement_results::H
end

"""
    SplitResult

Completed output of a [`SplitRun`](@ref). Scientific measurement results remain
separate from the control-only [`SplitLog`](@ref).
"""
struct SplitResult{
        M<:PeriodicMPS,
        H<:MeasurementHistory,
    }
    mps::M
    measurement_results::H
    log::SplitLog
end

"""
    SplitSnapshot

Durable numerical state captured from a [`SplitRun`](@ref) at a safe site
boundary. Derived transfer matrices and contraction caches are intentionally
excluded and will be rebuilt by `restore_run`.
"""
struct SplitSnapshot{
        M<:PeriodicMPS,
        P<:SplitPendingIteration,
        L<:SplitLog,
        H<:MeasurementHistory,
        I<:SplitProblemIdentity,
    } <: AbstractAlgorithmSnapshot
    approximation::M
    pending_iteration::P
    log::L
    measurement_results::H
    identity::I
end

"""
    SplitSiteCompleted

Progress event emitted after one split site has been updated successfully.
"""
struct SplitSiteCompleted{L<:SiteLog} <: AlgorithmEvent
    log::L
end

function _validate_split_sweep(sweep::AbstractSweep, site_count::Int)
    expected = Base.OneTo(site_count)
    domain = sweep_domain(sweep)
    domain == expected || throw(DimensionMismatch(
        "sweep domain $domain does not match split site domain $expected"
    ))
    return nothing
end

function initialize_run(target::PeriodicMPS, algorithm::ALSSplit)
    approximation = initial_approximation(target, algorithm.ansatz)
    _validate_split_sweep(algorithm.sweep, length(approximation))
    context = _build_split_context(target, approximation, algorithm.environments)
    pending_iteration = iterate(algorithm.sweep)
    measurement_results = MeasurementHistory()
    return SplitRun{
        typeof(algorithm),
        typeof(context),
        SplitPendingIteration,
        typeof(measurement_results),
    }(
        algorithm,
        context,
        pending_iteration,
        SplitLog(),
        measurement_results,
    )
end

isfinished(run::SplitRun) = isnothing(run.pending_iteration)

function _copy_split_log(log::SplitLog)
    return SplitLog(copy(log.site_logs), log.duration, log.converged)
end

function _completes_sweep(
        state::SweepState,
        following_iteration::SplitPendingIteration,
    )
    return isnothing(following_iteration) ||
           first(following_iteration).completed_sweeps > state.completed_sweeps
end

function snapshot(run::SplitRun)
    # A snapshot must remain unchanged if the run continues. Copy every tensor,
    # rather than only copying the PeriodicMPS container, and detach the mutable
    # log vector from the live run.
    approximation = PeriodicMPS([
        copy(tensor) for tensor in run.context.approximation
    ])
    identity = _split_problem_identity(run.context.target, run.algorithm)
    return SplitSnapshot(
        approximation,
        run.pending_iteration,
        _copy_split_log(run.log),
        deepcopy(run.measurement_results),
        identity,
    )
end

function _replay_split_progress(
        site_logs::AbstractVector{<:SplitSiteLog},
        sweep::AbstractSweep,
        stopping::AbstractStoppingCriterion,
    )
    expected_iteration = iterate(sweep)
    for (step, site_log) in enumerate(site_logs)
        isnothing(expected_iteration) && throw(ArgumentError(
            "snapshot contains a log for step $step after the supplied sweep is complete"
        ))
        expected_state, next_iterator_state = expected_iteration
        site_log.state == expected_state || throw(ArgumentError(
            "snapshot step $step has state $(site_log.state), expected $expected_state for the supplied sweep"
        ))
        following_iteration = iterate(sweep, next_iterator_state)
        if _completes_sweep(expected_state, following_iteration) &&
                should_stop(
                    stopping,
                    @view(site_logs[begin:step]),
                )
            step == length(site_logs) || throw(ArgumentError(
                "snapshot contains logs after the configured cost criterion converged"
            ))
            return nothing, true
        end
        expected_iteration = following_iteration
    end
    return expected_iteration, false
end

function _validate_split_cursor(
        saved::SplitSnapshot,
        sweep::AbstractSweep,
        stopping::AbstractStoppingCriterion,
    )
    expected_iteration, expected_converged = _replay_split_progress(
        saved.log.site_logs,
        sweep,
        stopping,
    )
    saved.log.converged == expected_converged || throw(ArgumentError(
        "snapshot convergence flag does not match its cost history"
    ))
    isequal(saved.pending_iteration, expected_iteration) || throw(ArgumentError(
        "snapshot cursor does not match its completed logs and the supplied sweep"
    ))
    return nothing
end

"""
    restore_run(saved::SplitSnapshot, target::PeriodicMPS, algorithm::ALSSplit)

Reconstruct a split run at the safe boundary captured by `saved`. Transfer
matrices and contraction caches are derived again from the immutable target
`target` and a private copy of the saved approximation.
"""
function restore_run(
        saved::SplitSnapshot,
        target::PeriodicMPS,
        algorithm::ALSSplit,
    )
    length(saved.approximation) == 2 * length(target) || throw(DimensionMismatch(
        "snapshot approximation has $(length(saved.approximation)) tensors " *
        "but the target has $(length(target)) two-physical-leg tensors " *
        "(expected $(2 * length(target)) approximation tensors)"
    ))
    _validate_split_sweep(algorithm.sweep, length(saved.approximation))
    _matches_split_problem(saved.identity, target, algorithm) || throw(
        ArgumentError(
            "snapshot does not match the supplied split target or algorithm configuration"
        )
    )
    _validate_split_cursor(saved, algorithm.sweep, algorithm.stopping)

    approximation = PeriodicMPS([
        copy(tensor) for tensor in saved.approximation
    ])
    context = _build_split_context(
        target, approximation, algorithm.environments
    )
    pending_iteration = saved.pending_iteration
    measurement_results = deepcopy(saved.measurement_results)
    return SplitRun{
        typeof(algorithm),
        typeof(context),
        SplitPendingIteration,
        typeof(measurement_results),
    }(
        algorithm,
        context,
        pending_iteration,
        _copy_split_log(saved.log),
        measurement_results,
    )
end

function step!(run::SplitRun)
    isfinished(run) && throw(ArgumentError("split run is already complete"))

    state, next_iterator_state = run.pending_iteration
    following_iteration = iterate(run.algorithm.sweep, next_iterator_state)

    started_at = time()
    _, cost = update!(
        run.context,
        run.algorithm.solver,
        run.algorithm.environments,
        state,
    )
    site_log = SiteLog(state, cost, time() - started_at)
    push!(run.log.site_logs, site_log)

    run.log.duration += site_log.duration
    run.pending_iteration = following_iteration
    if _completes_sweep(state, following_iteration) &&
            should_stop(run.algorithm.stopping, run.log.site_logs)
        run.log.converged = true
        run.pending_iteration = nothing
    end
    return SplitSiteCompleted(site_log)
end

function run_result(run::SplitRun)
    isfinished(run) || throw(ArgumentError("split run is not complete"))
    return SplitResult(
        run.context.approximation,
        run.measurement_results,
        run.log,
    )
end

"""
    split(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        algorithm::ALSSplit;
        observer,
    )

Given a periodic MPS with two physical legs per tensor, produce a periodic MPS
with one physical leg per tensor that best approximates the periodic state.
Returns `(approximation, log)` where `log` is a [`SplitLog`](@ref) with one
`SiteLog` per update. This blocking convenience API uses the same
[`SplitRun`](@ref) stepper as controllable local and platform execution.

Interactive sessions use the default split progress observer when `observer`
is omitted. Pass `observer=nothing` to disable output or supply a callable to
replace the default observer.
"""
function split(
        target::PeriodicMPS,
        algorithm::ALSSplit;
        observer = _DEFAULT_OBSERVER,
    )
    run = initialize_run(target, algorithm)
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
    result = run_result(run)
    return result.mps, result.log
end
