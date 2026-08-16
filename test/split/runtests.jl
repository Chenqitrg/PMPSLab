using Test
using Random
using Logging
using HalfIntegers: half
import ProgressLogging
import PMPSLab
using PMPSLab:
    PeriodicMPS,
    HalfInt,
    one_mps,
    random_cdl_mps,
    contract_pairs,
    dense,
    assertconsistency,
    scalartype,
    SplitContext,
    split,
    update!,
    initialize_run,
    step!,
    isfinished,
    run_result,
    snapshot,
    restore_run,
    ZigzagSweep,
    CircularSweep,
    Forward,
    SweepState,
    DenseTransferMatrix,
    TruncatedTransferMatrix,
    PeriodicTransferMatrix,
    AbstractTransferPolicy,
    DenseTransferPolicy,
    TruncatedTransferPolicy,
    StaleCacheError,
    environment,
    squared_distance,
    LUSolve,
    CholeskySolve,
    KrylovSolve,
    ALSSplit,
    AbstractStoppingCriterion,
    CostTolerance,
    CostConvergence,
    NormalizationResult,
    SplitEnvironmentPolicy,
    SplitLog,
    SplitRun,
    SplitResult,
    SplitSnapshot,
    SplitSiteCompleted,
    RandomAnsatz,
    OneAnsatz,
    SVDAnsatz,
    PeriodicGaugeAnsatz,
    default_looptnr
import PMPSLab: initial_approximation, split_tensor,
                current_approximation, paired_approximation, target_tensor, target_site_for, expectation,
                ProjectorGaugeFix, gaugefix!
using TensorKit
using KrylovKit: GMRES
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace,
              physicalspace, TruncationStrategy

include("ansatz_test.jl")
include("context_test.jl")
include("environment_test.jl")
include("transfer_policy_test.jl")
include("run_test.jl")
include("progress_test.jl")
include("stopping_test.jl")
include("snapshot_test.jl")
include("algorithm_test.jl")
include("periodic_gauge_test.jl")
