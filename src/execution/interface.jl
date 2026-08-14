"""
    AbstractAlgorithmRun

Mutable execution state of a resumable computation. Concrete runs coordinate
an algorithm configuration, an [`AlgorithmContext`](@ref), and progress such as
the next safe step and convergence log. A successful [`step!`](@ref) must leave
the run at a consistent snapshot boundary.
"""
abstract type AbstractAlgorithmRun end

"""
    AbstractAlgorithmSnapshot

Durable, platform-independent state sufficient to reconstruct an algorithm run
from the algorithm-specific immutable inputs and configuration.
"""
abstract type AbstractAlgorithmSnapshot end

"""
    AlgorithmEvent

Immutable description of one successfully completed algorithm step.
"""
abstract type AlgorithmEvent end

"""
    initialize_run(inputs..., algorithm) -> AbstractAlgorithmRun

Construct a fresh in-memory run. Concrete algorithms define their own input
arguments and dispatch on their algorithm configuration type.
"""
function initialize_run end

"""
    step!(run::AbstractAlgorithmRun) -> AlgorithmEvent

Advance `run` by one algorithm-defined safe step. A successful return marks a
consistent boundary at which the run may be inspected or snapshotted.
"""
function step! end

"""
    isfinished(run::AbstractAlgorithmRun) -> Bool

Return whether `run` has completed and its result is available.
"""
function isfinished end

"""
    run_result(run::AbstractAlgorithmRun)

Return the completed result of `run`. Concrete implementations should reject
calls made before [`isfinished`](@ref) becomes true.
"""
function run_result end

"""
    snapshot(run::AbstractAlgorithmRun) -> AbstractAlgorithmSnapshot

Capture the durable state required to restore `run`. Snapshots must exclude
runtime-only values such as tasks, locks, callbacks, and platform handles.
"""
function snapshot end

"""
    restore_run(snapshot::AbstractAlgorithmSnapshot, inputs..., algorithm) ->
        AbstractAlgorithmRun

Validate `snapshot` for compatibility with the algorithm-specific immutable
inputs and configuration, rebuild derived state, and return a run positioned
at exactly the next safe step.
"""
function restore_run end
