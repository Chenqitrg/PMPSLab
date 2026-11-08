"""
    SplitProgress

The latest completed site update, shared by state inspection, callbacks and logs.
Before the first update, counts are zero and site/cost fields are `nothing`.
`exact_error` belongs to this update only: it is measured at sweep boundaries
when `error_measure=:exact`, and is `nothing` on all other updates.
For NNR, costs are normalized; multiply by `cost_scale` for original-scale squared
errors. `penalty` is the xi used in this update, or `nothing` for ordinary ALS.
"""
Base.@kwdef struct SplitProgress <: ExperimentLab.AbstractLogEvent
    completed_updates::Int = 0
    completed_sweeps::Int = 0
    site::Union{Nothing,Int} = nothing
    sweep::Int = 0
    direction::Union{Nothing,String} = nothing
    local_cost::Union{Nothing,Float64} = nothing
    exact_error::Union{Nothing,Float64} = nothing
    sweep_completed::Bool = false
    converged::Bool = false
    duration::Float64 = 0.0
    penalty::Union{Nothing,Float64} = nothing
    cost_scale::Float64 = 1.0
end

const SplitPendingIteration = Union{Nothing,Tuple{LatticeIteratorLab.SweepState,Int}}

"""
    SplitState

Numerical workspace: tensors, caches, traversal and stopping-rule memory.
`progress` is the latest completed update. ExperimentLab owns the session lifecycle.
"""
mutable struct SplitState{A<:ALSSplit,T<:PeriodicMPS,M<:PeriodicMPS,NC,MC,E,S,R} <: AbstractSplitState
    algorithm::A
    target::T
    approximation::M
    normal_cache::NC
    mixed_cache::MC
    target_squared_norm::E
    sweep::S
    pending_iteration::SplitPendingIteration
    progress::SplitProgress
    last_error::Union{Nothing,Float64}
    stable_comparisons::Int
    regularization::R
end

# Each two-physical-leg target tensor maps to two approximation tensors.
target_site_for(site::Int) = cld(site, 2)
paired_approximation(state::SplitState, site::Int) =
    iseven(site) ? state.approximation[site - 1] : state.approximation[site + 1]

function _validate_initial(target::PeriodicMPS, initial::PeriodicMPS)
    all(t -> numout(t) == 3 && numin(t) == 1, target) ||
        throw(ArgumentError("target tensors must have two physical legs"))
    all(t -> numout(t) == 2 && numin(t) == 1, initial) ||
        throw(ArgumentError("initial tensors must have one physical leg"))
    length(initial) == 2length(target) || throw(DimensionMismatch(
        "initial MPS must have twice as many sites as the target"))
    collect(Iterators.flatten(physicalspaces(target))) ==
        collect(physicalspaces(initial)) || throw(DimensionMismatch(
            "initial physical spaces must match the target in pair order"))
    scalartype(target) == scalartype(initial) || throw(ArgumentError(
        "initial and target tensors must have the same scalar type"))
    return nothing
end

function initialize_split(target::PeriodicMPS, algorithm::ALSSplit)
    initial = initial_approximation(target, algorithm.initialization)
    return initialize_split(target, initial, algorithm; copy_initial=false)
end

"""
    initialize_split(target, initial, algorithm; copy_initial=true)

Create numerical state from an existing approximation, preserving its virtual
spaces. Copy every initial tensor by default. Set `copy_initial=false` to replace
tensors in the caller's MPS; do not modify it externally during optimization.
The target is read-only during optimization. Local solves and caches
still allocate memory in either mode.
"""
function initialize_split(
        target::PeriodicMPS,
        initial::PeriodicMPS,
        algorithm::ALSSplit;
        copy_initial::Bool=true,
    )
    _validate_initial(target, initial)
    approximation = copy_initial ? PeriodicMPS([copy(t) for t in initial]) : initial
    target_squared_norm = expectation(target, target)
    regularization = _initialize_regularization(algorithm.regularization, approximation, target_squared_norm)
    return _build_split_state(target, approximation, algorithm, target_squared_norm, regularization)
end

# Shared by fresh runs and restoration; restoring supplies saved NNR buffers
# directly instead of allocating zero buffers that would immediately be discarded.
function _build_split_state(
        target::PeriodicMPS, approximation::PeriodicMPS, algorithm::ALSSplit,
        target_squared_norm::Number, regularization,
    )
    sweep = LatticeIteratorLab.sweep_states(algorithm.sweep, length(approximation))
    normal_cache = build_transfer_cache(
        approximation, approximation, algorithm.environments.normal,
    )
    mixed_cache = build_transfer_cache(
        approximation, target, algorithm.environments.mixed,
    )
    return SplitState(
        algorithm,
        target,
        approximation,
        normal_cache,
        mixed_cache,
        target_squared_norm,
        sweep,
        iterate(sweep),
        SplitProgress(),
        nothing, # No previous sweep error.
        0,       # No stable comparisons yet.
        regularization,
    )
end
