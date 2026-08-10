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
end

"""
    SplitSnapshot

Durable numerical state captured from a [`SplitRun`](@ref) at a safe site
boundary. Derived transfer matrices and contraction caches are intentionally
excluded and will be rebuilt by `restore_run`.
"""
struct SplitSnapshot{M,P,L<:SplitLog} <: AbstractAlgorithmSnapshot
    mpsB::M
    pending_iteration::P
    log::L
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
    context = SplitContext(mps, approximation)
    pending_iteration = iterate(algorithm.sweep)
    return SplitRun{
        typeof(algorithm),typeof(context),SplitPendingIteration
    }(
        algorithm,
        context,
        pending_iteration,
        SplitLog(),
    )
end

isfinished(run::SplitRun) = isnothing(run.pending_iteration)

function _copy_split_log(log::SplitLog)
    return SplitLog(copy(log.site_logs), log.sweep_time, log.converged)
end

function snapshot(run::SplitRun)
    # A snapshot must remain unchanged if the run continues. Copy every tensor,
    # rather than only copying the PeriodicMPS container, and detach the mutable
    # log vector from the live run.
    mpsB = PeriodicMPS([copy(tensor) for tensor in run.context.mpsB])
    return SplitSnapshot(mpsB, run.pending_iteration, _copy_split_log(run.log))
end

function _validate_split_cursor(saved::SplitSnapshot, sweep::AbstractSweep)
    expected_iteration = iterate(sweep)
    for (step, site_log) in enumerate(saved.log.site_logs)
        isnothing(expected_iteration) && throw(ArgumentError(
            "snapshot contains a log for step $step after the supplied sweep is complete"
        ))
        expected_state, next_iterator_state = expected_iteration
        site_log.state == expected_state || throw(ArgumentError(
            "snapshot step $step has state $(site_log.state), expected $expected_state for the supplied sweep"
        ))
        expected_iteration = iterate(sweep, next_iterator_state)
    end
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
    _validate_split_cursor(saved, algorithm.sweep)

    mpsB = PeriodicMPS([copy(tensor) for tensor in saved.mpsB])
    context = SplitContext(mpsA, mpsB)
    pending_iteration = saved.pending_iteration
    return SplitRun{
        typeof(algorithm),typeof(context),SplitPendingIteration
    }(
        algorithm,
        context,
        pending_iteration,
        _copy_split_log(saved.log),
    )
end

function step!(run::SplitRun)
    isfinished(run) && throw(ArgumentError("split run is already complete"))

    state, next_iterator_state = run.pending_iteration
    following_iteration = iterate(run.algorithm.sweep, next_iterator_state)

    started_at = time()
    _, cost = update!(run.context, run.algorithm.solve, state)
    site_log = SiteLog(state, cost, time() - started_at)
    push!(run.log.site_logs, site_log)

    run.log.sweep_time += site_log.time
    run.pending_iteration = following_iteration
    return SplitSiteCompleted(site_log)
end

function run_result(run::SplitRun)
    isfinished(run) || throw(ArgumentError("split run is not complete"))
    return run.context.mpsB, run.log
end

"""
    split(mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, alg::PeriodicSplit)

Given a periodic MPS with 2 physical legs per site (3 codim legs), produce a
periodic MPS with 1 physical leg per site (2 codim legs) that best approximates
the periodic state. Returns `(mps_1site, log)` where `log` is a [`SplitLog`](@ref)
with one `SiteLog` per update. This blocking convenience API uses the same
[`SplitRun`](@ref) stepper as controllable local and platform execution.
"""
function split(mps::PeriodicMPS, algorithm::PeriodicSplit)
    run = initialize_run(mps, algorithm)
    while !isfinished(run)
        step!(run)
    end
    return run_result(run)
end
