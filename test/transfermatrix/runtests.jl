using Test
using Random
import PMPSLab
using PMPSLab:
    PeriodicMPS,
    one_mps,
    pair_contraction,
    dense,
    combine_truncation,
    DenseTransferMatrix,
    TruncatedTransferMatrix,
    PeriodicTransferMatrix,
    DenseTransferPolicy,
    TruncatedTransferPolicy,
    FormalIdentity,
    transfer_at,
    Forward,
    SweepState,
    build_cache,
    advance!,
    rebuild_left!,
    rebuild_right!,
    rebuild!,
    edge_space,
    edge_identity,
    expectation,
    squared_norm,
    squared_distance,
    diff_norm,
    normalize,
    normalize!,
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
