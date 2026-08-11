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
    LocalRunSession(run; observer, threaded=Threads.nthreads(:default) > 1)

Lightweight local adapter for controlling an [`AbstractAlgorithmRun`](@ref)
from a Julia REPL. It provides cooperative, in-memory pause and cancellation at
algorithm-defined safe-step boundaries; it is not a production job scheduler.

When `observer` is omitted, interactive sessions use the algorithm's default
observer. Pass `observer=nothing` to disable output or supply a callable to
replace the default observer.

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
        observer = _DEFAULT_OBSERVER,
        threaded::Bool = Threads.nthreads(:default) > 1,
    )
    resolved_observer = _resolve_observer(run, observer)
    return LocalRunSession(
        run,
        resolved_observer,
        threaded,
        ReentrantLock(),
        LOCAL_READY,
        nothing,
        nothing,
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
    should_stop, transition = lock(session.lock) do
        if session.status == LOCAL_PAUSE_REQUESTED
            session.status = LOCAL_PAUSED
            return true, LOCAL_PAUSED
        elseif session.status == LOCAL_CANCEL_REQUESTED
            session.status = LOCAL_CANCELLED
            return true, LOCAL_CANCELLED
        elseif session.status != LOCAL_RUNNING
            return true, nothing
        end
        return false, nothing
    end
    if transition == LOCAL_PAUSED
        _observer_paused!(session.observer, session.run)
    elseif transition == LOCAL_CANCELLED
        _observer_cancelled!(session.observer, session.run)
    end
    return should_stop
end

function _notify_observer(session::LocalRunSession, event::AlgorithmEvent)
    _observer_step!(session.observer, event, session.run)
    return nothing
end

function _drive_local_session!(session::LocalRunSession)
    lock(session.lock) do
        session.task = current_task()
    end
    try
        _observer_running!(session.observer, session.run)
        while true
            _finish_if_requested!(session) && return session

            if isfinished(session.run)
                _set_status!(session, LOCAL_COMPLETED)
                _observer_completed!(session.observer, session.run)
                return session
            end

            event = step!(session.run)
            _notify_observer(session, event)

            # Completion wins over a pause or cancellation request made while
            # the final step was executing.
            if isfinished(session.run)
                _set_status!(session, LOCAL_COMPLETED)
                _observer_completed!(session.observer, session.run)
                return session
            end

            # On a single-threaded REPL this is the point at which an already
            # entered command gets a chance to run before the next safe step.
            yield()
        end
    catch err
        lock(session.lock) do
            session.failure = CapturedException(err, catch_backtrace())
            session.status = LOCAL_FAILED
        end
        _observer_failed!(session.observer, session.run, err)
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

"""
    fetch(session::LocalRunSession)

Wait for the current driver task and return the completed algorithm result. A
failed run throws its captured exception with the original worker backtrace;
any nonterminal or controlled stop is rejected with its current status.
"""
function Base.fetch(session::LocalRunSession)
    wait(session)
    status, failure = lock(session.lock) do
        session.status, session.failure
    end

    status == LOCAL_COMPLETED && return run_result(session.run)
    status == LOCAL_FAILED && throw(failure)
    throw(ArgumentError(
        "local run did not complete; current status is $status"
    ))
end
