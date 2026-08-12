"""
    AbstractNormalization <: AbstractAlgorithmConfig

Configuration for an in-place tensor-network normalization.
"""
abstract type AbstractNormalization <: AbstractAlgorithmConfig end

"""
    InfinityNormNormalization <: AbstractNormalization

Normalize the independently stored gates of a tensor cell by the geometric
mean of their coefficient infinity norms.
"""
struct InfinityNormNormalization <: AbstractNormalization end

"""
    NormalizationResult(scale, log_scale)

Scientific result of a normalization. `scale` is the factor divided out of
each independently stored gate and `log_scale` is its natural logarithm.
"""
struct NormalizationResult{T<:Real} <: AbstractMeasurementResult
    scale::T
    log_scale::T
end

_stored_gates(cell::UniformGates) = (cell.gate,)
_stored_gates(cell::CheckerboardGates) = (cell.black, cell.white)
_stored_gates(cell::PlaquetteGates) = (
    cell.northwest,
    cell.northeast,
    cell.southwest,
    cell.southeast,
)

function _independent_gates(
        cell::AbstractTensorCell{G}
    ) where {G<:GenericGateTensor}
    stored = _stored_gates(cell)
    gates = G[]
    seen = IdDict{Any,Nothing}()
    for gate in stored
        haskey(seen, gate) && continue
        seen[gate] = nothing
        push!(gates, gate)
    end
    return gates
end

function _normalization_scale(
        gates::AbstractVector{<:GenericGateTensor}
    )
    norms = map(gate -> norm(gate, Inf), gates)
    for gate_norm in norms
        iszero(gate_norm) && throw(ArgumentError(
            "cannot normalize a tensor cell containing a zero gate"
        ))
        isfinite(gate_norm) || throw(DomainError(
            gate_norm,
            "cannot normalize a tensor cell containing a non-finite gate",
        ))
    end

    log_scale = sum(log, norms) / length(norms)
    isfinite(log_scale) || throw(DomainError(
        log_scale, "non-finite logarithmic tensor-cell normalization scale"
    ))

    scale = exp(log_scale)
    isfinite(scale) && !iszero(scale) || throw(DomainError(
        scale, "invalid tensor-cell normalization scale"
    ))
    inverse_scale = inv(scale)
    isfinite(inverse_scale) || throw(DomainError(
        inverse_scale, "non-finite inverse tensor-cell normalization scale"
    ))
    return scale, log_scale, inverse_scale
end

"""
    normalize!(cell::AbstractTensorCell, normalization::InfinityNormNormalization)

Normalize `cell` in place and return a [`NormalizationResult`](@ref). Aliased
gate fields are included and scaled exactly once. All norms and scale factors
are validated before any gate is modified.
"""
function LinearAlgebra.normalize!(
        cell::AbstractTensorCell,
        ::InfinityNormNormalization,
    )
    gates = _independent_gates(cell)
    scale, log_scale, inverse_scale = _normalization_scale(gates)
    foreach(gate -> rmul!(gate, inverse_scale), gates)
    return NormalizationResult(scale, log_scale)
end
