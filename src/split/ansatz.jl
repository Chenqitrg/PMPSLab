"""
    RandomAnsatz(; virtual_spaces)

Initialize the split approximation with random tensors on the supplied
site-aligned virtual spaces.
"""
@kwdef struct RandomAnsatz{V<:GradedSpace} <: AbstractAnsatz
    virtual_spaces::HalfPeriodicVector{V}
end

RandomAnsatz(Vs::AbstractVector{V}) where {V<:GradedSpace} =
    RandomAnsatz(HalfPeriodicVector(Vs))

function _ansatz_physicalspaces(
        target::PeriodicMPS, virtual_spaces::HalfPeriodicVector
    )
    expected_length = 2 * length(target)
    length(virtual_spaces) == expected_length || throw(DimensionMismatch(
        "ansatz has $(length(virtual_spaces)) virtual spaces; expected $expected_length"
    ))
    return PeriodicVector(collect(Iterators.flatten(physicalspaces(target))))
end

function initial_approximation(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, ansatz::RandomAnsatz
    ) where {E,S}
    P = _ansatz_physicalspaces(target, ansatz.virtual_spaces)
    return Random.rand(PeriodicMPS, E, P, ansatz.virtual_spaces)
end

"""
    OneAnsatz(; virtual_spaces)

Initialize the split approximation with identity-like tensors on the supplied
site-aligned virtual spaces.
"""
@kwdef struct OneAnsatz{V<:GradedSpace} <: AbstractAnsatz
    virtual_spaces::HalfPeriodicVector{V}
end

OneAnsatz(Vs::AbstractVector{V}) where {V<:GradedSpace} =
    OneAnsatz(HalfPeriodicVector(Vs))

function initial_approximation(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, ansatz::OneAnsatz
    ) where {E,S}
    P = _ansatz_physicalspaces(target, ansatz.virtual_spaces)
    return one_mps(E, P, ansatz.virtual_spaces)
end

"""
    SVDAnsatz(; trunc=TruncationStrategy())

Initialize each pair of split tensors from a local singular-value
decomposition using `trunc`.
"""
@kwdef struct SVDAnsatz{T<:TruncationStrategy} <: AbstractAnsatz
    trunc::T = TruncationStrategy()
end

function split_tensor(
        tensor::AbstractTensorMap{E,S,3,1}, trunc::TruncationStrategy
    ) where {E,S}
    # L ⊗ P₁ ⊗ P₂ ← R → repartition as (L ⊗ P₁) ← (R ⊗ P₂) → SVD.
    U, Sv, Vp, _ = svd_trunc(
        repartition(tensor, 2, 2), DefaultAlgorithm(; trunc)
    )
    sqS = sqrt(Sv)
    # Two single-physical-leg tensors: (L ⊗ P₁ ← χ), (χ ⊗ P₂ ← R).
    return U * sqS, repartition(sqS * Vp, 2, 1)
end

function initial_approximation(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, ansatz::SVDAnsatz
    ) where {E,S}
    tensors = [
        part for tensor in target for part in split_tensor(tensor, ansatz.trunc)
    ]
    return PeriodicMPS(tensors)
end

"""
    PeriodicGaugeAnsatz(;
        initial_split=SVDAnsatz(),
        gauge_fix=ProjectorGaugeFix(),
    )

Apply an SVD initialization followed by periodic projector gauge fixing.
"""
@kwdef struct PeriodicGaugeAnsatz{
        S<:SVDAnsatz,
        C<:AbstractGaugeFix,
    } <: AbstractAnsatz
    initial_split::S = SVDAnsatz()
    gauge_fix::C = ProjectorGaugeFix()
end

function initial_approximation(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        ansatz::PeriodicGaugeAnsatz,
    ) where {E,S}
    approximation = initial_approximation(target, ansatz.initial_split)
    gaugefix!(approximation, ansatz.gauge_fix)
    return approximation
end
