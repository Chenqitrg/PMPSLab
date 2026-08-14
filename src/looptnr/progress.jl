mutable struct LoopTNRProgressObserver
    progress_id::Union{Nothing,UUID}
    rg::Union{Nothing,Int}
    sweep::Union{Nothing,Int}
    sweep_time::Float64
    started::Bool
end

LoopTNRProgressObserver() =
    LoopTNRProgressObserver(nothing, nothing, nothing, 0.0, false)

_default_observer(::LoopTNRRun) = LoopTNRProgressObserver()

function _emit_progress!(
        observer::LoopTNRProgressObserver,
        name::String;
        fraction::Union{Nothing,Float64} = nothing,
        done::Bool = false,
    )
    id = isnothing(observer.progress_id) ? uuid4() : observer.progress_id
    observer.progress_id = done ? nothing : id
    displayed_fraction = done ? 1.0 : fraction
    progress = ProgressLogging.Progress(id, displayed_fraction; name, done)
    @logmsg ProgressLogging.ProgressLevel progress
    return nothing
end

function _close_progress!(observer::LoopTNRProgressObserver, name::String)
    isnothing(observer.progress_id) || _emit_progress!(observer, name; done=true)
    return nothing
end

function _progress_number(value::Real)
    return string(round(value; sigdigits=6))
end

function _observer_running!(observer::LoopTNRProgressObserver, run::LoopTNRRun)
    action = observer.started ? "resumed" : "started"
    observer.started = true
    @info "LoopTNR $action · $(_looptnr_position(run))"
    return nothing
end

function _observer_step!(
        observer::LoopTNRProgressObserver,
        event::LoopTNRSplitStepCompleted,
        run::LoopTNRRun,
    )
    site_event = event.event
    state = site_event.log.state
    sweep = state.n + 1
    if observer.rg != event.rg
        _close_progress!(observer, "LoopTNR RG changed")
        observer.rg = event.rg
        observer.sweep = nothing
    end
    if observer.sweep != sweep
        observer.sweep = sweep
        observer.sweep_time = 0.0
    end

    observer.sweep_time += site_event.log.time
    site_count = length(sweep_domain(run.algorithm.split.sweep))
    progress_name = "LoopTNR RG $(event.rg)/$(run.algorithm.maximal_rg) · " *
        "split sweep $sweep · site $(state.site)/$site_count · " *
        "cost $(_progress_number(site_event.log.error))"
    active_split = something(run.active_split)
    completed_updates = length(active_split.log.site_logs)
    maximal_updates = length(active_split.algorithm.sweep)
    fraction = maximal_updates == 0 ? nothing :
        min(completed_updates / maximal_updates, 1.0)
    _emit_progress!(observer, progress_name; fraction)

    if _completes_sweep(state, active_split.pending_iteration)
        summary = "LoopTNR RG $(event.rg)/$(run.algorithm.maximal_rg) · " *
            "split sweep $sweep completed · " *
            "cost $(_progress_number(site_event.log.error)) · " *
            "time $(_progress_number(observer.sweep_time)) s"
        @info summary
        isfinished(active_split) && _close_progress!(observer, progress_name)
        observer.sweep = nothing
        observer.sweep_time = 0.0
    end
    return nothing
end

function _rg_summary(event::LoopTNRCoarseGrained, run::LoopTNRRun)
    prefix = "LoopTNR RG $(event.rg)/$(run.algorithm.maximal_rg)"
    scale = _progress_number(event.result.normalization.scale)
    event.rg == 0 && return "$prefix initialized · scale $scale"

    split_log = event.log.split_log
    parts = String[prefix * " completed"]
    if !isempty(split_log.site_logs)
        push!(parts, "cost $(_progress_number(last(split_log.site_logs).error))")
    end
    push!(parts, "split $(_progress_number(split_log.sweep_time)) s")
    push!(parts, "coarse grain $(_progress_number(event.log.coarse_grain_time)) s")
    push!(parts, "scale $scale")
    return join(parts, " · ")
end

function _observer_step!(
        observer::LoopTNRProgressObserver,
        event::LoopTNRCoarseGrained,
        run::LoopTNRRun,
    )
    _close_progress!(observer, "LoopTNR RG $(event.rg) completed")
    observer.rg = event.rg
    observer.sweep = nothing
    observer.sweep_time = 0.0
    @info _rg_summary(event, run)
    return nothing
end

function _observer_paused!(observer::LoopTNRProgressObserver, run::LoopTNRRun)
    _close_progress!(observer, "LoopTNR paused")
    @info "LoopTNR paused · $(_looptnr_position(run))"
    return nothing
end

function _observer_cancelled!(observer::LoopTNRProgressObserver, run::LoopTNRRun)
    _close_progress!(observer, "LoopTNR cancelled")
    @info "LoopTNR cancelled · $(_looptnr_position(run))"
    return nothing
end

function _observer_completed!(observer::LoopTNRProgressObserver, run::LoopTNRRun)
    _close_progress!(observer, "LoopTNR completed")
    @info "LoopTNR completed · RG $(run.completed_rg)/$(run.algorithm.maximal_rg) · " *
        "$(length(run.measurement_results)) measurements"
    return nothing
end

function _observer_failed!(
        observer::LoopTNRProgressObserver,
        run::LoopTNRRun,
        exception,
    )
    _close_progress!(observer, "LoopTNR failed")
    @error "LoopTNR failed · $(_looptnr_position(run)) · " *
        sprint(showerror, exception)
    return nothing
end
