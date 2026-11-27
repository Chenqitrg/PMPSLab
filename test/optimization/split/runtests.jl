using PMPSLab: SplitEnvironmentPolicy
using LatticeIteratorLab
using LatticeIteratorLab: SweepState, Forward, Backward
using PeriodicProductsLab
using TensorLinearSolveLab
using Test, Random, Logging
using PMPSLab
import AlgorithmsInterface as AI
using PMPSLab: SplitProgress, AbstractStoppingCriterion
using PMPSLab: update!
using PeriodicProductsLab
using TensorKit
using KrylovKit: GMRES
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace,
              physicalspace, TruncationStrategy
import PMPSLab: initial_approximation, split_tensor,
    paired_approximation, target_site_for

# Small numerical fixtures share the same explicit initialization path as users.
function test_split_state(target, approximation;
                          solver=CholeskySolve(), environments=SplitEnvironmentPolicy())
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=1), solver=solver,
                         environments=environments)
    return PMPSLab.initialize_split(target, approximation, algorithm; copy_initial=false)
end

# Test consumer of the public A.I. protocol. Production execution uses AI.solve.
function run_split!(state::PMPSLab.SplitState; onstep::F=Returns(nothing))::NamedTuple{(:mps,:progress)} where {F}
    problem, algorithm = PMPSLab.SplitProblem(state.target), state.algorithm
    while !AI.is_finished!(problem, algorithm, state)
        AI.increment!(state)
        AI.step!(problem, algorithm, state)
        onstep(state, state.progress)
    end
    return AI.finalize_state!(problem, algorithm, state)
end

function collect_split(target::PeriodicMPS, algorithm::ALSSplit)::Tuple{NamedTuple,Vector{SplitProgress}}
    events = SplitProgress[]
    state = PMPSLab.initialize_split(target, algorithm)
    result = run_split!(state; onstep=function collect_progress(::PMPSLab.SplitState, event::SplitProgress)::Nothing
        push!(events, event)
        return nothing
    end)
    return result, events
end

include("ansatz_test.jl")
include("workspace_test.jl")
include("cache_boundary_test.jl")
include("environment_test.jl")
include("transfer_policy_test.jl")
include("state_test.jl")
include("stopping_test.jl")
include("algorithm_test.jl")
include("periodic_gauge_test.jl")

include("cost_regression_test.jl")

include("nuclear_norm_test.jl")
include("precision_performance_test.jl")
