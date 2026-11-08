using PMPSLab: SplitEnvironmentPolicy
using LatticeIteratorLab
using LatticeIteratorLab: SweepState, Forward, Backward
using PeriodicProductsLab
using TensorLinearSolveLab
using Test, Random, Logging
using PMPSLab, ExperimentLab
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

# History collection is a test consumer of onstep, not part of numerical state.
function collect_split(target, algorithm)
    events = SplitProgress[]
    state = PMPSLab.initialize_split(target, algorithm)
    result = fetch(start_run(ExperimentLab.algorithm_steps(state);
        onstep=(_, event) -> push!(events, event), threaded=false))
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
