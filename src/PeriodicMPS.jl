module PeriodicMPS

using TensorKit
import TensorKit: scalartype
using KrylovKit
using MPSKit: left_virtualspace, right_virtualspace, physicalspace,
              Algorithm, GenericMPSTensor, TruncationStrategy, PeriodicVector

# Types
export PeriodicMPS, HalfInt
export ConvergenceLog, PowerIterLog, SweepLog, ProjectorLog, ApproximationLog
export AbstractLoopCanonical, ProjectorCanonical, NuclearNormCanonical
export AbstractLoopApproximation
export LoopTransferMatrix
export RingCache, build_cache, environment, left_cache, right_cache

# Public API
export canonicalize!, assertconsistency, dense, pair_contraction, approximate
export random_mps, one_mps, random_mps2, one_mps2
export AbstractSweep, ZipSweep, CircularSweep, ApproximationContext, update!

include("canonicalize/halfint.jl")
include("mps.jl")
include("logs.jl")
include("circularupdate/loopmonoid.jl")
include("circularupdate/cache.jl")
include("transfermatrix.jl")
include("canonicalize/projector.jl")
include("canonicalize/nuclearnorm.jl")
include("canonicalize/interface.jl")
include("assert.jl")
include("approximate/interface.jl")
include("circularupdate/sweep.jl")
include("approximate/context.jl")
include("dense.jl")

end
