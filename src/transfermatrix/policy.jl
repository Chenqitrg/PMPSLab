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
abstract type AbstractTransferPolicy <: AbstractAlgorithmConfig end

"""
    DenseTransferPolicy()

Construct exact, materialized [`DenseTransferMatrix`](@ref) values.
"""
struct DenseTransferPolicy <: AbstractTransferPolicy end

"""
    TruncatedTransferPolicy(truncation)

Construct factored [`TruncatedTransferMatrix`](@ref) site values and apply
`truncation` again whenever transfer products are accumulated.
"""
struct TruncatedTransferPolicy{
        T<:TruncationStrategy
    } <: AbstractTransferPolicy
    truncation::T
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
) = transfer_at(bra, ket, site, policy.truncation)

build_cache(
    bra::PeriodicMPS, ket::PeriodicMPS, ::DenseTransferPolicy
) = build_cache(bra, ket)
build_cache(
    bra::PeriodicMPS, ket::PeriodicMPS, policy::TruncatedTransferPolicy
) = build_cache(bra, ket, policy.truncation)

function rebuild_left!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        ::DenseTransferPolicy,
    )
    return rebuild_left!(cache, bra, ket)
end
function rebuild_left!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        policy::TruncatedTransferPolicy,
    )
    return rebuild_left!(cache, bra, ket, policy.truncation)
end

function rebuild_right!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        ::DenseTransferPolicy,
    )
    return rebuild_right!(cache, bra, ket)
end
function rebuild_right!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        policy::TruncatedTransferPolicy,
    )
    return rebuild_right!(cache, bra, ket, policy.truncation)
end

function rebuild!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        ::DenseTransferPolicy,
    )
    return rebuild!(cache, bra, ket)
end
function rebuild!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        policy::TruncatedTransferPolicy,
    )
    return rebuild!(cache, bra, ket, policy.truncation)
end
