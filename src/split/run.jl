"""
    SplitRun

Mutable execution state for a [`PeriodicSplit`](@ref). `context` owns the
numerical working state, while the run owns the algorithm, sweep cursor,
and progress log.
"""
mutable struct SplitRun{M,A<:PeriodicSplit,C<:SplitContext,P} <: AbstractAlgorithmRun
    target::M
    algorithm::A
    context::C
    pending_iteration::Union{Nothing,P}
    log::SplitLog
end

"""
    SplitSiteCompleted

Progress event emitted after one split site has been updated successfully.
"""
struct SplitSiteCompleted{L<:SiteLog} <: AlgorithmEvent
    log::L
end

function initialize_run(mps::PeriodicMPS, algorithm::PeriodicSplit)
    approximation = initial_approximation(mps, algorithm.ansatz)
    context = SplitContext(mps, approximation)
    pending_iteration = iterate(algorithm.sweep)
    return SplitRun{
        typeof(mps),typeof(algorithm),typeof(context),typeof(pending_iteration)
    }(
        mps,
        algorithm,
        context,
        pending_iteration,
        SplitLog(),
    )
end

isfinished(run::SplitRun) = isnothing(run.pending_iteration)

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
