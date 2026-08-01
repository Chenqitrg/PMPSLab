struct RandomAnsatz{I<:Sector} <: AbstractAnsatz
    virtual_space::HalfPeriodicVector{<:GradedSpace{I}}
end

RandomAnsatz(V::GradedSpace{I}, N::Int) where {I<:Sector} = RandomAnsatz{I}(fill(V, N))

function initial_approximation(mps::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}, ansatz::RandomAnsatz{I}) where {E, S<:ElementarySpace{I}, I}
    N = 2 * length(mps)
    @assert length(ansatz.virtual_space) == N "ansatz virtual space length must match mps length"
    # physicalspaces(mps) returns ProductSpace elements — flatten each collect → vcat
    P_spaces = PeriodicVector(reduce(vcat, collect.(physicalspaces(mps))))
    return random_mps(E, P_spaces, ansatz.virtual_space)
end

struct OneAnsatz{I<:Sector} <: AbstractAnsatz
    virtual_space::HalfPeriodicVector{<:GradedSpace{I}}
end

OneAnsatz(V::GradedSpace{I}, N::Int) where {I<:Sector} = OneAnsatz{I}(fill(V, N))

function initial_approximation(mps::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}, ansatz::OneAnsatz{I}) where {E, S<:ElementarySpace{I}, I}
    N = 2 * length(mps)
    @assert length(ansatz.virtual_space) == N "ansatz virtual space length must match mps length"
    # physicalspaces(mps) returns ProductSpace elements — flatten each collect → vcat
    P_spaces = PeriodicVector(reduce(vcat, collect.(physicalspaces(mps))))
    return one_mps(E, P_spaces, ansatz.virtual_space)
end

struct SVDAnsatz <: AbstractAnsatz
    truncation::TruncationStrategy
end

function split_site(T::AbstractTensorMap{E, S, 3, 1}, trunc::TruncationStrategy) where {E, S}
    # T: L ⊗ P₁ ⊗ P₂ ← R  →  transpose to (L ⊗ P₁) ← (R ⊗ P₂)  → SVD
    U, S, Vp, _ = svd_trunc(transpose(T, ((1, 2), (4, 3))), DefaultAlgorithm(; trunc))
    sqS = sqrt(S)
    return U * sqS, transpose(sqS * Vp, ((1, 3), (2,)))      # two 1-site tensors: (L ⊗ P₁ ← χ), (χ ⊗ P₂ ← R)
end

function initial_approximation(mps::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}, ansatz::SVDAnsatz) where {E, S}
    tensors = [t for i in eachindex(mps) for t in split_site(mps[i], ansatz.truncation)]
    return PeriodicMPS(tensors)
end