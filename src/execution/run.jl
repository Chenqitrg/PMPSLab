const SPLIT_EVENTS = ExperimentLab.EventRegistry("mps.split.update.v1" => SplitProgress)

# Keep Logging's macro temporaries outside the resumable numerical function.
function _log_split_progress(event::SplitProgress)
    record = ExperimentLab.event_record(SPLIT_EVENTS, event)
    @info "Split site completed" record...
    return nothing
end

"""
    split(target, algorithm)
    split(target, initial, algorithm; copy_initial=true)

Run a split to completion and return a named tuple `(mps=..., progress=...)`. Both this
entry point and controlled runs use ExperimentLab's generator driver. For stored
logs and callbacks, pass `algorithm_steps(initialize_split(...))` to `start_run`.
An explicit initial MPS fixes the virtual spaces and is copied by default.
"""
function split(target::PeriodicMPS, algorithm::AbstractSplit)
    state = initialize_split(target, algorithm)
    session = ExperimentLab.start_run(algorithm_steps(state); threaded=false)
    return fetch(session)
end

function split(
        target::PeriodicMPS,
        initial::PeriodicMPS,
        algorithm::AbstractSplit;
        copy_initial::Bool=true,
    )
    state = initialize_split(target, initial, algorithm; copy_initial)
    session = ExperimentLab.start_run(algorithm_steps(state); threaded=false)
    return fetch(session)
end
