using PMPSLab: rebuild_left_transfer_cache!, rebuild_right_transfer_cache!
using Test, Random, PMPSLab, PeriodicProductsLab
using TensorKit
using MPSKit: AbstractTransferMatrix, PeriodicVector, TransferMatrix,
              TruncationStrategy, left_virtualspace
# Manual pairwise overlap via dense states. One-site bra and ket physical legs
# have the same positional ordering.
function overlap(bra::PeriodicMPS, ket::PeriodicMPS)
    return scalar(dense(bra)' * dense(ket))
end

include("periodic_test.jl")
include("distance_test.jl")

include("cache_integration_test.jl")

include("package_boundary_test.jl")

include("cache_inputs_test.jl")

include("configuration_test.jl")
