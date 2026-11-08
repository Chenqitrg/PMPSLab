"""Configuration for splitting each two-physical-leg MPS site into a pair of sites."""
abstract type AbstractSplit <: ExperimentLab.AbstractAlgorithmConfig end

"""Algorithm-owned numerical workspace. ExperimentLab owns execution status."""
abstract type AbstractSplitState end

"""
    initialize_split(target, algorithm::AbstractSplit)

Create an algorithm's AbstractSplitState. The target is read-only. Algorithms
may also support an explicit initial approximation as a second argument.
"""
function initialize_split end

"""
    split_steps(state::AbstractSplitState)

Return a ResumableFunctions generator yielding algorithm-owned AbstractLogEvent
values at safe pause/capture boundaries. Return a named tuple containing `mps`,
the optimized paired single-physical-leg PeriodicMPS. Additional result fields
are algorithm-specific. Each implementation logs its own events.
"""
function split_steps end

"""
    restore_split(snapshot, target; algorithm)

Restore the state captured by ExperimentLab.copy_state at a yielded boundary.
Validate the supplied configuration against the snapshot and rebuild workspaces
from the read-only target. The next generator must not replay completed updates.
Implementations may default `algorithm` to a configuration stored in the snapshot.
"""
function restore_split end
