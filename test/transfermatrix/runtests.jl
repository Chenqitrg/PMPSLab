using Test
using Random
import PMPSLab
using PMPSLab:
    PeriodicMPS,
    random_mps,
    one_mps,
    random_mps2,
    pair_contraction,
    dense,
    combine_truncation,
    DenseTransferMatrix,
    TruncatedTransferMatrix,
    PeriodicTransferMatrix,
    build_cache,
    advance!,
    SweepState,
    edge_identity,
    expectation,
    squared_norm,
    squared_distance,
    diff_norm,
    gaugefix!,
    ProjectorCanonical
using TensorKit
import MatrixAlgebraKit
using MPSKit: AbstractTransferMatrix, PeriodicVector, TransferMatrix,
              TruncationStrategy, left_virtualspace

# Manual pairwise overlap via dense states. One-site bra and ket physical legs
# have the same positional ordering.
function overlap(bra::PeriodicMPS, ket::PeriodicMPS)
    return scalar(dense(bra)' * dense(ket))
end

include("dense_test.jl")
include("truncated_test.jl")
include("periodic_test.jl")
include("distance_test.jl")
