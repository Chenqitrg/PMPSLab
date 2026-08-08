"""
    SplitRun

Mutable execution state for a [`PeriodicSplit`](@ref). `context` owns the
numerical working state, while the run owns the algorithm, sweep cursor,
and progress log.
"""
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

function initialize_run(mps::PeriodicMPS, algorithm::PeriodicSplit)
    approximation = initial_approximation(mps, algorithm.ansatz)
    context = SplitContext(mps, approximation)
    pending_iteration = iterate(algorithm.sweep)
    return SplitRun{
        typeof(algorithm),typeof(context),typeof(pending_iteration)
    }(
        algorithm,
        context,
        pending_iteration,
        SplitLog(),
    )
end

isfinished(run::SplitRun) = isnothing(run.pending_iteration)

function snapshot(run::SplitRun)
    # A snapshot must remain unchanged if the run continues. Copy every tensor,
    # rather than only copying the PeriodicMPS container, and detach the mutable
    # log vector from the live run.
    mpsB = PeriodicMPS([copy(tensor) for tensor in run.context.mpsB])
    log = SplitLog(
        copy(run.log.site_logs),
        run.log.sweep_time,
        run.log.converged,
    )
    return SplitSnapshot(mpsB, run.pending_iteration, log)
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
