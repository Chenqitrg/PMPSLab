# ------------------------------------------------------------------------------
# Transfer representation policies
# ------------------------------------------------------------------------------

"""
    AbstractTransferPolicy

Algorithm-level policy selecting how transfer matrices and their accumulated
products are represented. The policy is separate from the transfer value type:
`DenseTransferMatrix` and `TruncatedTransferMatrix` contain numerical data,
while a policy constructs and refreshes values of the chosen representation.
"""
abstract type AbstractTransferPolicy <: ExperimentLab.AbstractAlgorithmConfig end

"""
    DenseTransferPolicy()

Construct exact, materialized [`DenseTransferMatrix`](@ref) values.
"""
@option struct DenseTransferPolicy <: AbstractTransferPolicy end

"""
    TruncatedTransferPolicy(; trunc=truncerror(rtol=1e-13))

Construct factored [`TruncatedTransferMatrix`](@ref) site values and apply
`trunc` again whenever transfer products are accumulated.
"""
@option struct TruncatedTransferPolicy{
        T<:MPSKit.TruncationStrategy
    } <: AbstractTransferPolicy
    trunc::T = TensorKit.truncerror(; rtol=1.0e-13)
end

transfer_at(
    mps::PeriodicMPS, site::Int, policy::AbstractTransferPolicy
) = transfer_at(mps, mps, site, policy)
transfer_at(
    bra::PeriodicMPS, ket::PeriodicMPS, site::Int, ::DenseTransferPolicy
) = transfer_at(bra, ket, site)
transfer_at(
    bra::PeriodicMPS, ket::PeriodicMPS, site::Int,
    policy::TruncatedTransferPolicy,
) = transfer_at(bra, ket, site, policy.trunc)

build_transfer_cache(
    bra::PeriodicMPS, ket::PeriodicMPS, ::DenseTransferPolicy
) = build_transfer_cache(bra, ket)
build_transfer_cache(
    bra::PeriodicMPS, ket::PeriodicMPS, policy::TruncatedTransferPolicy
) = build_transfer_cache(bra, ket, policy.trunc)

function rebuild_left_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        ::DenseTransferPolicy,
    )
    return rebuild_left_transfer_cache!(cache, bra, ket)
end
function rebuild_left_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        policy::TruncatedTransferPolicy,
    )
    return rebuild_left_transfer_cache!(cache, bra, ket, policy.trunc)
end

function rebuild_right_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        ::DenseTransferPolicy,
    )
    return rebuild_right_transfer_cache!(cache, bra, ket)
end
function rebuild_right_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        policy::TruncatedTransferPolicy,
    )
    return rebuild_right_transfer_cache!(cache, bra, ket, policy.trunc)
end

function rebuild_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        ::DenseTransferPolicy,
    )
    return rebuild_transfer_cache!(cache, bra, ket)
end
function rebuild_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        policy::TruncatedTransferPolicy,
    )
    return rebuild_transfer_cache!(cache, bra, ket, policy.trunc)
end
