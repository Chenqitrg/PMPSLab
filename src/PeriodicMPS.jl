module PeriodicMPS

using TensorKit
using KrylovKit
using MPSKit: left_virtualspace, right_virtualspace, physicalspace

# Types
export PeriodicMPS, HalfInt
export ConvergenceLog, ProjectorLog, ApproximationLog
export AbstractLoopCanonical, ProjectorCanonical, NuclearNormCanonical
export AbstractLoopApproximation
export LoopTransferMatrix

# Public API
export canonicalize!, assertconsistency, dense, approximate

include("types.jl")
include("transfermatrix.jl")
include("canonicalize/interface.jl")
include("assert.jl")
include("canonicalize/projector.jl")
include("canonicalize/nuclearnorm.jl")
include("dense.jl")
include("approximate/interface.jl")

end
