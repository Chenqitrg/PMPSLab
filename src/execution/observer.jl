struct DefaultObserver end

const _DEFAULT_OBSERVER = DefaultObserver()

_default_observer(::AbstractAlgorithmRun) = nothing

function _resolve_observer(
        run::AbstractAlgorithmRun,
        observer,
        interactive::Bool = isinteractive(),
    )
    observer === _DEFAULT_OBSERVER || return observer
    return interactive ? _default_observer(run) : nothing
end

_observer_running!(observer, run::AbstractAlgorithmRun) = nothing
_observer_paused!(observer, run::AbstractAlgorithmRun) = nothing
_observer_cancelled!(observer, run::AbstractAlgorithmRun) = nothing
_observer_completed!(observer, run::AbstractAlgorithmRun) = nothing
_observer_failed!(observer, run::AbstractAlgorithmRun, exception) = nothing

_observer_step!(::Nothing, event::AlgorithmEvent, run::AbstractAlgorithmRun) = nothing
function _observer_step!(observer, event::AlgorithmEvent, run::AbstractAlgorithmRun)
    observer(event)
    return nothing
end
