module PMPSLab

import Base: split
using TensorKit
import TensorKit: scalartype
import JLD2
using KrylovKit
using LinearAlgebra
import MatrixAlgebraKit
using MPSKit: left_virtualspace, right_virtualspace,
              AbstractTransferMatrix, Algorithm, GenericMPSTensor,
              TruncationStrategy, PeriodicVector, DefaultAlgorithm
import MPSKit
import MPSKit: physicalspace

# Types
export PeriodicMPS, HalfInt, HalfPeriodicVector, physicalspaces
export ConvergenceLog, SiteLog, SweepLog, SplitLog
export AbstractAlgorithmConfig
export AbstractAlgorithmRun, AbstractAlgorithmSnapshot
export AlgorithmEvent, LocalRunSession, LocalRunStatus, LocalCheckpointOutcome
export AbstractPeriodicCanonical, ProjectorCanonical
export AbstractAnsatz, AbstractSolve, AbstractPeriodicSplit
export SplitEnvironmentPolicy
export PeriodicSplit, SplitRun, SplitSnapshot, SplitSiteCompleted
export RandomAnsatz, OneAnsatz, SVDAnsatz, PeriodicGaugeAnsatz
export DenseTransferMatrix, TruncatedTransferMatrix, PeriodicTransferMatrix
export AbstractTransferPolicy, DenseTransferPolicy, TruncatedTransferPolicy
export transfer_at
export AbstractPeriodicProduct, PeriodicCache, build_cache, environment
export prefix_products, suffix_products
export left_cache_before, right_cache_after, range_product, edge_space, edge_identity
export FormalIdentity, UnitalElement, compose_product
export StaleCacheError
export invalidate!, advance!, rebuild_left!, rebuild_right!, rebuild!

# Public API
export gaugefix!, assertconsistency, dense, pair_contraction, split
export initialize_run, step!, isfinished, run_result, snapshot, restore_run
export write_checkpoint, read_checkpoint
export start_run, start_run!, pause!, resume!, cancel!, run_status
export checkpoint!, pause_and_checkpoint!, cancel_and_checkpoint!
export squared_norm, squared_distance, diff_norm
export expectation
export random_mps, one_mps, random_mps2, one_mps2, random_cdl, random_cdl_mps, lift
export SweepDirection, Forward, Backward, SweepState
export AbstractSweep, ZigzagSweep, CircularSweep, AlgorithmContext, SplitContext, update!
export LUSolve, LDLTSolve, CholeskySolve, KrylovSolve, ldiv, rdiv

include("algorithm_config.jl")
include("execution/interface.jl")
include("execution/local_session.jl")
include("execution/local_control.jl")
include("mps/indices.jl")
include("mps/container.jl")
include("mps/constructors.jl")
include("mps/cdl.jl")
include("mps/contraction.jl")
include("mps/lift.jl")
include("circularupdate/periodicproduct.jl")
include("circularupdate/formalidentity.jl")
include("circularupdate/productscans.jl")
include("logs.jl")
include("circularupdate/cache.jl")
include("circularupdate/generatedproducts.jl")
include("circularupdate/advance.jl")
include("transfermatrix/dense.jl")
include("transfermatrix/truncation.jl")
include("transfermatrix/truncated.jl")
include("transfermatrix/periodic.jl")
include("transfermatrix/policy.jl")
include("transfermatrix/distance.jl")
include("canonicalize/projector.jl")
include("canonicalize/interface.jl")
include("split/interface.jl")
include("split/ansatz.jl")
include("circularupdate/sweep.jl")
include("solve.jl")
include("algorithm_context.jl")
include("split/environment.jl")
include("split/context.jl")
include("split/identity.jl")
include("split/run.jl")
include("execution/checkpoint.jl")
include("execution/local_checkpoint.jl")

end
