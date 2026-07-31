module PeriodicMPS

using TensorKit
using KrylovKit
using MPSKit: left_virtualspace, right_virtualspace, physicalspace

# Types
export PeriodicMPS, HalfInt
export ConvergenceLog, ProjectorLog
export AbstractLoopCanonical, ProjectorCanonical, NuclearNormCanonical

# Public API
export canonicalize!, assertconsistency, dense

include("types.jl")
include("canonicalize/interface.jl")
include("assert.jl")
include("canonicalize/projector.jl")
include("canonicalize/nuclearnorm.jl")
include("dense.jl")

end
