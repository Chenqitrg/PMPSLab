"""
    RandomAnsatz(; virtual_space)

Initialize the split approximation with random tensors on the supplied
site-aligned virtual spaces.
"""
@kwdef struct RandomAnsatz{V<:GradedSpace} <: AbstractAnsatz
    virtual_space::HalfPeriodicVector{V}
end

RandomAnsatz(Vs::AbstractVector{V}) where {V<:GradedSpace} =
    RandomAnsatz(HalfPeriodicVector(Vs))

function _ansatz_physicalspaces(mps::PeriodicMPS, virtual_space::HalfPeriodicVector)
    expected_length = 2 * length(mps)
    length(virtual_space) == expected_length || throw(DimensionMismatch(
        "ansatz has $(length(virtual_space)) virtual spaces; expected $expected_length"
    ))
    return PeriodicVector(collect(Iterators.flatten(physicalspaces(mps))))
end

function initial_approximation(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, ansatz::RandomAnsatz
    ) where {E,S}
    P = _ansatz_physicalspaces(mps, ansatz.virtual_space)
    return Random.rand(PeriodicMPS, E, P, ansatz.virtual_space)
end

"""
    OneAnsatz(; virtual_space)

Initialize the split approximation with identity-like tensors on the supplied
site-aligned virtual spaces.
"""
@kwdef struct OneAnsatz{V<:GradedSpace} <: AbstractAnsatz
    virtual_space::HalfPeriodicVector{V}
end

OneAnsatz(Vs::AbstractVector{V}) where {V<:GradedSpace} =
    OneAnsatz(HalfPeriodicVector(Vs))

function initial_approximation(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, ansatz::OneAnsatz
    ) where {E,S}
    P = _ansatz_physicalspaces(mps, ansatz.virtual_space)
    return one_mps(E, P, ansatz.virtual_space)
end

"""
    SVDAnsatz(; trunc=TruncationStrategy())

Initialize each pair of split tensors from a local singular-value
decomposition using `trunc`.
"""
@kwdef struct SVDAnsatz{T<:TruncationStrategy} <: AbstractAnsatz
    trunc::T = TruncationStrategy()
end

function split_site(T::AbstractTensorMap{E, S, 3, 1}, trunc::TruncationStrategy) where {E, S}
    # T: L ⊗ P₁ ⊗ P₂ ← R  →  repartition as (L ⊗ P₁) ← (R ⊗ P₂)  → SVD
    U, Sv, Vp, _ = svd_trunc(repartition(T, 2, 2), DefaultAlgorithm(; trunc))
    sqS = sqrt(Sv)
    return U * sqS, repartition(sqS * Vp, 2, 1)      # two 1-site tensors: (L ⊗ P₁ ← χ), (χ ⊗ P₂ ← R)
end

function initial_approximation(mps::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}, ansatz::SVDAnsatz) where {E, S}
    tensors = [tensor for site in mps for tensor in split_site(site, ansatz.trunc)]
    return PeriodicMPS(tensors)
end

"""
    PeriodicGaugeAnsatz(;
        svd_pre=SVDAnsatz(),
        canonical=ProjectorCanonical(),
    )

Apply an SVD initialization followed by periodic projector gauge fixing.
"""
@kwdef struct PeriodicGaugeAnsatz{
        S<:SVDAnsatz,
        C<:AbstractPeriodicCanonical,
    } <: AbstractAnsatz
    svd_pre::S = SVDAnsatz()
    canonical::C = ProjectorCanonical()
end

function initial_approximation(mps::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}, ansatz::PeriodicGaugeAnsatz) where {E, S}
    mps_approx = initial_approximation(mps, ansatz.svd_pre)
    gaugefix!(mps_approx, ansatz.canonical)
    return mps_approx
end
