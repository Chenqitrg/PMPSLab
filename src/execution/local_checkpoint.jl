"""
    LocalCheckpointOutcome

Result of a local durable operation. `LOCAL_CHECKPOINT_WRITTEN` confirms that
the requested file was published. `LOCAL_CHECKPOINT_SKIPPED_COMPLETED` means a
final algorithm step completed before cancellation took effect, so no file was
created.
"""
@enum LocalCheckpointOutcome::UInt8 begin
    LOCAL_CHECKPOINT_WRITTEN
    LOCAL_CHECKPOINT_SKIPPED_COMPLETED
end

function _reject_driver_wait(session::LocalRunSession, operation::AbstractString)
    task = lock(session.lock) do
        session.task
    end
    current_task() === task && throw(ArgumentError(
        "$operation cannot be called from the session observer"
    ))
    return nothing
end

function _request_checkpoint_cancel!(session::LocalRunSession)
    lock(session.lock) do
        if session.status == LOCAL_FAILED
            throw(ArgumentError("cannot cancel and checkpoint failed local run"))
        elseif session.status == LOCAL_COMPLETED
            return false
        elseif session.status == LOCAL_RUNNING ||
               session.status == LOCAL_PAUSE_REQUESTED
            session.status = LOCAL_CANCEL_REQUESTED
        elseif session.status == LOCAL_READY || session.status == LOCAL_PAUSED
            session.status = LOCAL_CANCELLED
        end
        return true
    end
end

"""
    checkpoint!(session::LocalRunSession, path::AbstractString) ->
        LocalCheckpointOutcome

Write a checkpoint for a quiescent paused or cancelled session. The live run
remains available in memory after writing, so a paused session can still be
resumed without rebuilding derived caches.
"""
function checkpoint!(session::LocalRunSession, path::AbstractString)
    saved = lock(session.lock) do
        status = session.status
        status == LOCAL_PAUSED || status == LOCAL_CANCELLED || throw(
            ArgumentError(
                "cannot checkpoint local run from status $status; pause or cancel it first"
            )
        )
        snapshot(session.run)
    end
    write_checkpoint(path, saved)
    return LOCAL_CHECKPOINT_WRITTEN
end

"""
    pause_and_checkpoint!(session::LocalRunSession, path::AbstractString) ->
        LocalCheckpointOutcome

Request a cooperative pause, wait until the current safe step has completed,
then write a checkpoint. The function returns only after the session is paused
and the checkpoint has been published. A completed, cancelled, or failed run
is rejected without writing a checkpoint.
"""
function pause_and_checkpoint!(session::LocalRunSession, path::AbstractString)
    _reject_driver_wait(session, "pause_and_checkpoint!")
    status = run_status(session)
    status == LOCAL_RUNNING || status == LOCAL_PAUSE_REQUESTED ||
        status == LOCAL_PAUSED || throw(ArgumentError(
            "cannot pause and checkpoint local run from status $status"
        ))

    pause!(session)
    wait(session)
    run_status(session) == LOCAL_PAUSED || throw(ArgumentError(
        "local run did not pause; current status is $(run_status(session))"
    ))
    return checkpoint!(session, path)
end

"""
    cancel_and_checkpoint!(session::LocalRunSession, path::AbstractString) ->
        LocalCheckpointOutcome

Request cooperative cancellation, wait for the current safe step, then write a
checkpoint of the cancelled state. If the final step completes before the
cancellation request takes effect, the completed session is returned without
writing a checkpoint.
"""
function cancel_and_checkpoint!(session::LocalRunSession, path::AbstractString)
    _reject_driver_wait(session, "cancel_and_checkpoint!")
    _request_checkpoint_cancel!(session) ||
        return LOCAL_CHECKPOINT_SKIPPED_COMPLETED
    wait(session)

    status = run_status(session)
    status == LOCAL_COMPLETED && return LOCAL_CHECKPOINT_SKIPPED_COMPLETED
    status == LOCAL_CANCELLED || throw(ArgumentError(
        "local run did not cancel; current status is $status"
    ))
    return checkpoint!(session, path)
end
