abstract type AbstractInitialization <: ExperimentLab.AbstractAlgorithmConfig end

"""
    RandomInitialization(; virtual_spaces)

Initialize the split approximation with random tensors on the supplied
right virtual spaces: `virtual_spaces[i] == right_virtualspace(mps, i)`.
"""
Configurations.@option struct RandomInitialization{V<:ElementarySpace} <: AbstractInitialization
    virtual_spaces::MPSKit.PeriodicVector{V}
end

RandomInitialization(Vs::AbstractVector{V}) where {V<:ElementarySpace} =
    RandomInitialization(MPSKit.PeriodicVector(Vs))

function _initial_physicalspaces(
        target::PeriodicMPS, virtual_spaces::MPSKit.PeriodicVector
    )
    expected_length = 2 * length(target)
    length(virtual_spaces) == expected_length || throw(DimensionMismatch(
        "initialization has $(length(virtual_spaces)) virtual spaces; expected $expected_length"
    ))
    return MPSKit.PeriodicVector(collect(Iterators.flatten(physicalspaces(target))))
end

function initial_approximation(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, initialization::RandomInitialization
    ) where {E,S}
    P = _initial_physicalspaces(target, initialization.virtual_spaces)
    return Random.rand(PeriodicMPS, E, P, initialization.virtual_spaces)
end

"""
    IdentityInitialization(; virtual_spaces)

Initialize the split approximation with identity-like tensors on the supplied
right virtual spaces: `virtual_spaces[i] == right_virtualspace(mps, i)`.
"""
Configurations.@option struct IdentityInitialization{V<:ElementarySpace} <: AbstractInitialization
    virtual_spaces::MPSKit.PeriodicVector{V}
end

IdentityInitialization(Vs::AbstractVector{V}) where {V<:ElementarySpace} =
    IdentityInitialization(MPSKit.PeriodicVector(Vs))

function initial_approximation(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, initialization::IdentityInitialization
    ) where {E,S}
    P = _initial_physicalspaces(target, initialization.virtual_spaces)
    return one_mps(E, P, initialization.virtual_spaces)
end

"""
    SVDInitialization(; trunc=TruncationStrategy())

Initialize each pair of split tensors from a local singular-value
decomposition using `trunc`.
"""
Configurations.@option struct SVDInitialization{T<:MPSKit.TruncationStrategy} <: AbstractInitialization
    trunc::T = MPSKit.TruncationStrategy()
end

function split_tensor(
        tensor::AbstractTensorMap{E,S,3,1}, trunc::MPSKit.TruncationStrategy
    ) where {E,S}
    # L ⊗ P₁ ⊗ P₂ ← R → repartition as (L ⊗ P₁) ← (R ⊗ P₂) → SVD.
    U, Sv, Vp, _ = svd_trunc(
        repartition(tensor, 2, 2), MPSKit.DefaultAlgorithm(; trunc)
    )
    sqS = sqrt(Sv)
    # Two single-physical-leg tensors: (L ⊗ P₁ ← χ), (χ ⊗ P₂ ← R).
    return U * sqS, repartition(sqS * Vp, 2, 1)
end

function initial_approximation(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, initialization::SVDInitialization
    ) where {E,S}
    tensors = [
        part for tensor in target for part in split_tensor(tensor, initialization.trunc)
    ]
    return PeriodicMPS(tensors)
end

"""
    GaugeInitialization(;
        initialization=SVDInitialization(),
        gauge_fix=ProjectorGaugeFix(),
    )

Apply an initialization followed by periodic projector gauge fixing.
"""
Configurations.@option struct GaugeInitialization{
        S<:AbstractInitialization,
        C<:AbstractGaugeFix,
    } <: AbstractInitialization
    initialization::S = SVDInitialization()
    gauge_fix::C = ProjectorGaugeFix()
end

function initial_approximation(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        initialization::GaugeInitialization,
    ) where {E,S}
    approximation = initial_approximation(target, initialization.initialization)
    gaugefix!(approximation, initialization.gauge_fix)
    return approximation
end
