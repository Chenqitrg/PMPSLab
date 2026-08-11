@enum LocalRunStatus::UInt8 begin
    LOCAL_READY
    LOCAL_RUNNING
    LOCAL_PAUSE_REQUESTED
    LOCAL_PAUSED
    LOCAL_CANCEL_REQUESTED
    LOCAL_CANCELLED
    LOCAL_COMPLETED
    LOCAL_FAILED
end

"""
    LocalRunSession(run; observer=nothing, threaded=Threads.nthreads(:default) > 1)

Lightweight local adapter for controlling an [`AbstractAlgorithmRun`](@ref)
from a Julia REPL. It provides cooperative, in-memory pause and cancellation at
algorithm-defined safe-step boundaries; it is not a production job scheduler.

With more than one default Julia thread, `threaded=true` executes the run on a
worker thread so the REPL remains responsive during a step. With one thread,
the session yields after every completed step, so a command already entered at
the REPL is handled before a later step begins.
"""
mutable struct LocalRunSession{R<:AbstractAlgorithmRun,O}
    run::R
    observer::O
    threaded::Bool
    lock::ReentrantLock
    status::LocalRunStatus
    task::Union{Nothing,Task}
    failure::Any
end

function LocalRunSession(
        run::AbstractAlgorithmRun;
        observer = nothing,
        threaded::Bool = Threads.nthreads(:default) > 1,
    )
    return LocalRunSession(
        run, observer, threaded, ReentrantLock(), LOCAL_READY, nothing, nothing
    )
end

"""
    run_status(session::LocalRunSession) -> LocalRunStatus

Return a synchronized snapshot of the local session state.
"""
function run_status(session::LocalRunSession)
    lock(session.lock) do
        return session.status
    end
end

function _set_status!(session::LocalRunSession, status::LocalRunStatus)
    lock(session.lock) do
        session.status = status
    end
    return session
end

function _finish_if_requested!(session::LocalRunSession)
    lock(session.lock) do
        if session.status == LOCAL_PAUSE_REQUESTED
            session.status = LOCAL_PAUSED
            return true
        elseif session.status == LOCAL_CANCEL_REQUESTED
            session.status = LOCAL_CANCELLED
            return true
        elseif session.status != LOCAL_RUNNING
            return true
        end
        return false
    end
end

function _notify_observer(session::LocalRunSession, event::AlgorithmEvent)
    isnothing(session.observer) || session.observer(event)
    return nothing
end

function _drive_local_session!(session::LocalRunSession)
    lock(session.lock) do
        session.task = current_task()
    end
    try
        while true
            _finish_if_requested!(session) && return session

            if isfinished(session.run)
                _set_status!(session, LOCAL_COMPLETED)
                return session
            end

            event = step!(session.run)
            _notify_observer(session, event)

            # Completion wins over a pause or cancellation request made while
            # the final step was executing.
            if isfinished(session.run)
                _set_status!(session, LOCAL_COMPLETED)
                return session
            end

            # On a single-threaded REPL this is the point at which an already
            # entered command gets a chance to run before the next safe step.
            yield()
        end
    catch err
        lock(session.lock) do
            session.failure = (err, catch_backtrace())
            session.status = LOCAL_FAILED
        end
        return session
    end
end

function _launch_local_session!(session::LocalRunSession)
    lock(session.lock) do
        session.status == LOCAL_READY || session.status == LOCAL_PAUSED ||
            throw(ArgumentError(
                "cannot start local run from status $(session.status)"
            ))
        session.status = LOCAL_RUNNING
        session.failure = nothing
    end

    task = if session.threaded
        Threads.@spawn :default _drive_local_session!(session)
    else
        @async _drive_local_session!(session)
    end

    lock(session.lock) do
        session.task = task
    end
    return session
end

"""
    start_run!(session::LocalRunSession) -> LocalRunSession

Start or restart a local session and return immediately to the caller.
"""
start_run!(session::LocalRunSession) = _launch_local_session!(session)

"""
    start_run(run::AbstractAlgorithmRun; kwargs...) -> LocalRunSession

Create and start a [`LocalRunSession`](@ref).
"""
function start_run(run::AbstractAlgorithmRun; kwargs...)
    return start_run!(LocalRunSession(run; kwargs...))
end

function Base.wait(session::LocalRunSession)
    task = lock(session.lock) do
        session.task
    end
    isnothing(task) || wait(task)
    return session
end

function run_result(session::LocalRunSession)
    run_status(session) == LOCAL_COMPLETED || throw(ArgumentError(
        "local run has not completed; current status is $(run_status(session))"
    ))
    return run_result(session.run)
end
