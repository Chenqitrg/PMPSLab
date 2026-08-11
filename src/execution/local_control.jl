"""
    pause!(session::LocalRunSession) -> LocalRunSession

Request a cooperative pause. `run_status(session)` first becomes
`LOCAL_PAUSE_REQUESTED` and becomes `LOCAL_PAUSED` after the current safe step.
"""
function pause!(session::LocalRunSession)
    lock(session.lock) do
        if session.status == LOCAL_RUNNING
            session.status = LOCAL_PAUSE_REQUESTED
        elseif session.status != LOCAL_PAUSE_REQUESTED &&
               session.status != LOCAL_PAUSED
            throw(ArgumentError(
                "cannot pause local run from status $(session.status)"
            ))
        end
    end
    return session
end

"""
    resume!(session::LocalRunSession) -> LocalRunSession

Resume a paused run. If the pause has only been requested, cancel that request
without launching another task.
"""
function resume!(session::LocalRunSession)
    should_launch = lock(session.lock) do
        if session.status == LOCAL_PAUSE_REQUESTED
            session.status = LOCAL_RUNNING
            return false
        elseif session.status == LOCAL_PAUSED
            return true
        elseif session.status == LOCAL_RUNNING
            return false
        end
        throw(ArgumentError(
            "cannot resume local run from status $(session.status)"
        ))
    end
    should_launch && _launch_local_session!(session)
    return session
end

"""
    cancel!(session::LocalRunSession) -> LocalRunSession

Request cooperative cancellation without a checkpoint. Cancellation of a
running run takes effect after the current safe step; a paused run is cancelled
immediately.
"""
function cancel!(session::LocalRunSession)
    lock(session.lock) do
        if session.status == LOCAL_RUNNING ||
           session.status == LOCAL_PAUSE_REQUESTED
            session.status = LOCAL_CANCEL_REQUESTED
        elseif session.status == LOCAL_READY || session.status == LOCAL_PAUSED
            session.status = LOCAL_CANCELLED
        elseif session.status != LOCAL_CANCEL_REQUESTED &&
               session.status != LOCAL_CANCELLED
            throw(ArgumentError(
                "cannot cancel local run from status $(session.status)"
            ))
        end
    end
    return session
end
