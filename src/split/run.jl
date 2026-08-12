"""
    SplitRun

Mutable execution state for a [`PeriodicSplit`](@ref). `context` owns the
numerical working state, while the run owns the algorithm, sweep cursor,
and progress log.
"""
const SplitPendingIteration = Union{
    Nothing,
    Tuple{SweepState,Int},
}

mutable struct SplitRun{A<:PeriodicSplit,C<:SplitContext,P} <: AbstractAlgorithmRun
    algorithm::A
    context::C
    pending_iteration::Union{Nothing,P}
    log::SplitLog
    measurement_results::MeasurementHistory
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
        M,
        P,
        L<:SplitLog,
        I<:SplitProblemIdentity,
    } <: AbstractAlgorithmSnapshot
    mpsB::M
    pending_iteration::P
    log::L
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

function initialize_run(mps::PeriodicMPS, algorithm::PeriodicSplit)
    approximation = initial_approximation(mps, algorithm.ansatz)
    _validate_split_sweep(algorithm.sweep, length(approximation))
    context = _build_split_context(mps, approximation, algorithm.environments)
    pending_iteration = iterate(algorithm.sweep)
    return SplitRun{
        typeof(algorithm),typeof(context),SplitPendingIteration
    }(
        algorithm,
        context,
        pending_iteration,
        SplitLog(),
        MeasurementHistory(),
    )
end

isfinished(run::SplitRun) = isnothing(run.pending_iteration)

function _copy_split_log(log::SplitLog)
    return SplitLog(copy(log.site_logs), log.sweep_time, log.converged)
end

function _completes_sweep(
        state::SweepState,
        following_iteration::SplitPendingIteration,
    )
    return isnothing(following_iteration) ||
           first(following_iteration).n > state.n
end

function snapshot(run::SplitRun)
    # A snapshot must remain unchanged if the run continues. Copy every tensor,
    # rather than only copying the PeriodicMPS container, and detach the mutable
    # log vector from the live run.
    mpsB = PeriodicMPS([copy(tensor) for tensor in run.context.mpsB])
    identity = _split_problem_identity(run.context.mpsA, run.algorithm)
    return SplitSnapshot(
        mpsB,
        run.pending_iteration,
        _copy_split_log(run.log),
        identity,
    )
end

function _replay_split_progress(
        site_logs::AbstractVector{<:SplitSiteLog},
        sweep::AbstractSweep,
        stopping::Union{Nothing,AbstractStoppingCriterion},
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
        stopping::Union{Nothing,AbstractStoppingCriterion},
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
    restore_run(saved::SplitSnapshot, mpsA::PeriodicMPS, algorithm::PeriodicSplit)

Reconstruct a split run at the safe boundary captured by `saved`. Transfer
matrices and contraction caches are derived again from the immutable target
`mpsA` and a private copy of the saved approximation.
"""
function restore_run(
        saved::SplitSnapshot,
        mpsA::PeriodicMPS,
        algorithm::PeriodicSplit,
    )
    length(saved.mpsB) == 2 * length(mpsA) || throw(DimensionMismatch(
        "snapshot approximation has $(length(saved.mpsB)) sites but target has " *
        "$(length(mpsA)) two-site tensors (expected $(2 * length(mpsA)))"
    ))
    _validate_split_sweep(algorithm.sweep, length(saved.mpsB))
    _matches_split_problem(saved.identity, mpsA, algorithm) || throw(
        ArgumentError(
            "snapshot does not match the supplied split target or algorithm configuration"
        )
    )
    _validate_split_cursor(saved, algorithm.sweep, algorithm.stopping)

    mpsB = PeriodicMPS([copy(tensor) for tensor in saved.mpsB])
    context = _build_split_context(mpsA, mpsB, algorithm.environments)
    pending_iteration = saved.pending_iteration
    return SplitRun{
        typeof(algorithm),typeof(context),SplitPendingIteration
    }(
        algorithm,
        context,
        pending_iteration,
        _copy_split_log(saved.log),
        MeasurementHistory(),
    )
end

function step!(run::SplitRun)
    isfinished(run) && throw(ArgumentError("split run is already complete"))

    state, next_iterator_state = run.pending_iteration
    following_iteration = iterate(run.algorithm.sweep, next_iterator_state)

    started_at = time()
    _, cost = update!(
        run.context,
        run.algorithm.solve,
        run.algorithm.environments,
        state,
    )
    site_log = SiteLog(state, cost, time() - started_at)
    push!(run.log.site_logs, site_log)

    run.log.sweep_time += site_log.time
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
        run.context.mpsB,
        run.measurement_results,
        run.log,
    )
end

"""
    split(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        alg::PeriodicSplit;
        observer,
    )

Given a periodic MPS with 2 physical legs per site (3 codim legs), produce a
periodic MPS with 1 physical leg per site (2 codim legs) that best approximates
the periodic state. Returns `(mps_1site, log)` where `log` is a [`SplitLog`](@ref)
with one `SiteLog` per update. This blocking convenience API uses the same
[`SplitRun`](@ref) stepper as controllable local and platform execution.

Interactive sessions use the default split progress observer when `observer`
is omitted. Pass `observer=nothing` to disable output or supply a callable to
replace the default observer.
"""
function split(
        mps::PeriodicMPS,
        algorithm::PeriodicSplit;
        observer = _DEFAULT_OBSERVER,
    )
    run = initialize_run(mps, algorithm)
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
