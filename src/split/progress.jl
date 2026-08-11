mutable struct SplitProgressObserver
    progress_id::Union{Nothing,UUID}
    sweep::Union{Nothing,Int}
    sweep_time::Float64
    started::Bool
end

SplitProgressObserver() = SplitProgressObserver(nothing, nothing, 0.0, false)

_default_observer(::SplitRun) = SplitProgressObserver()

function _emit_progress!(
        observer::SplitProgressObserver,
        name::String;
        done::Bool = false,
    )
    id = isnothing(observer.progress_id) ? uuid4() : observer.progress_id
    observer.progress_id = done ? nothing : id
    fraction = done ? 1.0 : nothing
    progress = ProgressLogging.Progress(id, fraction; name, done)
    @logmsg ProgressLogging.ProgressLevel progress
    return nothing
end

function _close_progress!(observer::SplitProgressObserver, name::String)
    isnothing(observer.progress_id) || _emit_progress!(observer, name; done=true)
    return nothing
end

function _observer_running!(observer::SplitProgressObserver, ::SplitRun)
    if observer.started
        @info "Periodic split resumed"
    else
        observer.started = true
        @info "Periodic split started"
    end
    return nothing
end

function _observer_step!(
        observer::SplitProgressObserver,
        event::SplitSiteCompleted,
        run::SplitRun,
    )
    state = event.log.state
    sweep = state.n + 1
    if observer.sweep != sweep
        _close_progress!(observer, "Periodic split sweep changed")
        observer.sweep = sweep
        observer.sweep_time = 0.0
    end

    observer.sweep_time += event.log.time
    progress_name =
        "Periodic split: sweep=$sweep site=$(state.site) cost=$(event.log.error)"
    _emit_progress!(observer, progress_name)

    if _completes_sweep(state, run.pending_iteration)
        _close_progress!(observer, progress_name)
        summary = "Periodic split sweep $sweep completed: " *
            "cost=$(event.log.error), time=$(observer.sweep_time)s"
        @info summary
        observer.sweep = nothing
        observer.sweep_time = 0.0
    end
    return nothing
end

function _observer_paused!(observer::SplitProgressObserver, ::SplitRun)
    _close_progress!(observer, "Periodic split paused")
    @info "Periodic split paused"
    return nothing
end

function _observer_cancelled!(observer::SplitProgressObserver, ::SplitRun)
    _close_progress!(observer, "Periodic split cancelled")
    @info "Periodic split cancelled"
    return nothing
end

function _observer_completed!(observer::SplitProgressObserver, run::SplitRun)
    _close_progress!(observer, "Periodic split completed")
    @info "Periodic split completed after $(length(run.log.site_logs)) site updates"
    return nothing
end

function _observer_failed!(
        observer::SplitProgressObserver,
        ::SplitRun,
        exception,
    )
    _close_progress!(observer, "Periodic split failed")
    @error "Periodic split failed: $(sprint(showerror, exception))"
    return nothing
end
