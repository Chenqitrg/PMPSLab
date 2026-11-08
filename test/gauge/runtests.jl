using PMPSLab: P_decomp, find_projectors
using Test, Random, PMPSLab, TensorKit
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace
import MPSKit
include("interface_test.jl")
include("projector_test.jl")
include("gaugefix_test.jl")

include("configuration_test.jl")
