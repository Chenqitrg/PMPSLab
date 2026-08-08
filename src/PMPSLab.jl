module PMPSLab

import Base: split
using TensorKit
import TensorKit: scalartype
using KrylovKit
using LinearAlgebra
using MPSKit: left_virtualspace, right_virtualspace,
              Algorithm, GenericMPSTensor, TruncationStrategy, PeriodicVector,
              DefaultAlgorithm
import MPSKit
import MPSKit: physicalspace

# Types
export PeriodicMPS, HalfInt, HalfPeriodicVector, physicalspaces
export ConvergenceLog, SiteLog, SweepLog, SplitLog
export AbstractPeriodicCanonical, ProjectorCanonical
export AbstractAnsatz, AbstractSolve, AbstractPeriodicSplit
export PeriodicSplit, RandomAnsatz, OneAnsatz, SVDAnsatz, PeriodicGaugeAnsatz
export PeriodicTransferMatrix
export AbstractPeriodicProduct, PeriodicCache, build_cache, environment
export left_cache, right_cache, range_product, edge_identity

# Public API
export gaugefix!, assertconsistency, dense, pair_contraction, split
export squared_norm, squared_distance, diff_norm
export random_mps, one_mps, random_mps2, one_mps2, random_cdl, random_cdl_mps, lift
export SweepState, AbstractSweep, ZigzagSweep, CircularSweep, AlgorithmContext, SplitContext, update!
export LUSolve, LDLTSolve, CholeskySolve, KrylovSolve, ldiv, rdiv

include("mps.jl")
include("circularupdate/periodicproduct.jl")
include("logs.jl")
include("circularupdate/cache.jl")
include("transfermatrix.jl")
include("canonicalize/projector.jl")
include("canonicalize/interface.jl")
include("split/interface.jl")
include("split/ansatz.jl")
include("circularupdate/sweep.jl")
include("solve.jl")
include("algorithm_context.jl")
include("split/context.jl")
include("dense.jl")

end
