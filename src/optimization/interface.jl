"""Configuration for splitting each two-physical-leg MPS site into a pair of sites."""
abstract type AbstractSplit <: ExperimentLab.AbstractAlgorithmConfig end

"""
Algorithm-owned split workspace implementing `ExperimentLab.algorithm_steps`.
The generator yields algorithm-owned AbstractLogEvent values at safe pause/capture
boundaries and returns a named tuple containing `mps`, the optimized paired
single-physical-leg PeriodicMPS. Additional result fields are algorithm-specific.
Each implementation logs its own events; ExperimentLab owns execution control.
"""
abstract type AbstractSplitState <: ExperimentLab.AbstractAlgorithmState end

"""
    initialize_split(target, algorithm::AbstractSplit)

Create an algorithm's AbstractSplitState. The target is read-only. Algorithms
may also support an explicit initial approximation as a second argument.
"""
function initialize_split end

"""
    restore_split(snapshot, target; algorithm)

Restore the state captured by ExperimentLab.copy_state at a yielded boundary.
Validate the supplied configuration against the snapshot and rebuild workspaces
from the read-only target. The next generator must not replay completed updates.
Implementations may default `algorithm` to a configuration stored in the snapshot.
"""
function restore_split end
