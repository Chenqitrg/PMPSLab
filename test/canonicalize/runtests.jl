using Test
using Random
import PMPSLab
using PMPSLab:
    PeriodicMPS,
    HalfInt,
    HalfPeriodicVector,
    assertconsistency,
    gaugefix!,
    dense,
    pair_contraction,
    one_mps,
    one_mps2,
    ProjectorCanonical,
    P_decomp,
    find_projectors
using TensorKit
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace
import MPSKit

include("interface_test.jl")
include("consistency_test.jl")
include("projector_test.jl")
include("gaugefix_test.jl")
include("dense_one_site_test.jl")
include("pair_contraction_test.jl")
include("dense_two_site_test.jl")
