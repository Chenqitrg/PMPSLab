"""Configuration for splitting each two-physical-leg MPS site into a pair of sites."""
abstract type AbstractSplit <: AI.Algorithm end

"""
Numerical workspace implementing AlgorithmsInterface's state and step interfaces.
`iterate` is the current MPS; `iteration` counts the site update being performed
or last performed. `progress.completed_updates` counts successfully committed updates.
"""
abstract type AbstractSplitState <: AI.State end

"""Read-only target for a split. The algorithm supplies its initialization policy."""
struct SplitProblem{T<:PeriodicMPS} <: AI.Problem
    target::T
end

"""
    initialize_split(target, algorithm::AbstractSplit)

Create an algorithm's AbstractSplitState. The target is read-only. Algorithms
may also support an explicit initial approximation as a second argument.
"""
function initialize_split end

"""
    restore_split(snapshot, target; algorithm)

Restore the state captured by `snapshot` at a completed update boundary.
Validate the supplied configuration against the snapshot and rebuild workspaces
from the read-only target. Continuing must not replay completed updates.
Implementations may default `algorithm` to a configuration stored in the snapshot.
"""
function restore_split end
