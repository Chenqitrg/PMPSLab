using Test
using Random
using HalfIntegers: half
import PMPSLab
using PMPSLab:
    PeriodicMPS,
    HalfInt,
    HalfPeriodicVector,
    assertconsistency,
    gaugefix!,
    dense,
    contract_pairs,
    one_mps,
    ProjectorGaugeFix,
    P_decomp,
    find_projectors
using TensorKit
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace
import MPSKit

include("interface_test.jl")
include("consistency_test.jl")
include("projector_test.jl")
include("gaugefix_test.jl")
include("dense_single_physical_leg_test.jl")
include("contract_pairs_test.jl")
include("dense_two_physical_legs_test.jl")
