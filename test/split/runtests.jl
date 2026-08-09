using Test
using Random
import PMPSLab
using PMPSLab:
    PeriodicMPS,
    HalfInt,
    random_mps,
    one_mps,
    random_mps2,
    random_cdl_mps,
    pair_contraction,
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
    SweepState,
    DenseTransferMatrix,
    TruncatedTransferMatrix,
    PeriodicTransferMatrix,
    environment,
    squared_distance,
    LUSolve,
    CholeskySolve,
    KrylovSolve,
    PeriodicSplit,
    SplitRun,
    SplitSnapshot,
    SplitSiteCompleted,
    RandomAnsatz,
    OneAnsatz,
    SVDAnsatz,
    PeriodicGaugeAnsatz
import PMPSLab: initial_approximation, split_site, diff_norm,
                this_B, other_B, this_A, pa_from_pb, expectation,
                ProjectorCanonical, gaugefix!
using TensorKit
using KrylovKit: GMRES
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace,
              physicalspace, TruncationStrategy

include("ansatz_test.jl")
include("context_test.jl")
include("environment_test.jl")
include("run_test.jl")
include("snapshot_test.jl")
include("algorithm_test.jl")
include("periodic_gauge_test.jl")
