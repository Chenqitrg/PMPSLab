# Both arrays are in normalized units and have exactly the MPS tensor spaces.
struct NuclearNormState{T<:AbstractTensorMap}
    auxiliary::Vector{T}
    dual::Vector{T}
end

_initialize_regularization(::NoRegularization, mps::PeriodicMPS, squared_norm::Number) = nothing
function _initialize_regularization(::NuclearNorm, mps::PeriodicMPS, squared_norm::Number)
    _validate_nnr_norm(squared_norm)
    return NuclearNormState([zero(t) for t in mps], [zero(t) for t in mps])
end

function _validate_nnr_norm(squared_norm::Number)
    C = real(squared_norm)
    isfinite(C) && C > 0 || throw(ArgumentError("NNR requires a finite, nonzero target norm"))
    return nothing
end

_cost_scale(state::SplitState) =
    state.algorithm.regularization isa NuclearNorm ? real(state.target_squared_norm) : 1.0

# Normalized target A/s^2 and approximation B/s, where C=<A|A> and s=C^(1/(2n)).
# Scaling the local equation is equivalent and avoids copying the whole target.
function _site_scale(state::SplitState)
    C = real(state.target_squared_norm)
    return C^(one(C) / (2length(state.approximation)))
end

# Numerical kernels and progress use the same penalty rounded to tensor precision.
function _penalty(state::SplitState)
    xi = _penalty(state.algorithm.regularization, state.progress.completed_sweeps)
    isnothing(xi) && return nothing
    R = real(scalartype(state.approximation[1]))
    value = R(xi)
    isfinite(value) && value > 0 || throw(ArgumentError(
        "NNR penalty must be finite and positive in tensor precision $R; got $xi"))
    return value
end

function _regularized_solution(
        ::Nothing, state::SplitState, site::Int,
        initial::AbstractTensorMap, environment::MPSKit.AbstractTransferMatrix,
        rhs::AbstractTensorMap,
    )
    return _solve_split_norm(state.algorithm.solver, initial, environment, rhs)
end

function _regularized_solution(
        buffers::NuclearNormState, state::SplitState, site::Int,
        initial::AbstractTensorMap, environment::MPSKit.AbstractTransferMatrix,
        rhs::AbstractTensorMap,
    )
    C = real(state.target_squared_norm)
    s = _site_scale(state)
    xi = _penalty(state)
    auxiliary = transpose(buffers.auxiliary[site], ((2,), (1, 3)))
    dual = transpose(buffers.dual[site], ((2,), (1, 3)))
    normalized_rhs = (s / C) * rhs + xi * auxiliary + dual
    normalized_solution = _solve_split_norm(
        state.algorithm.solver, initial / s, environment, normalized_rhs;
        scale=s^2 / C, shift=xi,
    )
    return s * normalized_solution
end

# Work through TensorKit's sector blocks, without assuming a flat storage layout.
function _soft_threshold(tensor::AbstractTensorMap, threshold::Real)
    U, S, V, _ = svd_trunc(tensor, MPSKit.DefaultAlgorithm(; trunc=MPSKit.TruncationStrategy()))
    for (_, block) in blocks(S)
        block.diag .= max.(block.diag .- threshold, zero(eltype(block)))
    end
    return U * S * V
end

# Isolate the inserted bond: (L,P)|R at odd sites, L|(P,R) at even sites.
function _threshold_pair_bond(tensor::AbstractTensorMap, site::Int, threshold::Real)
    if isodd(site)
        return _soft_threshold(tensor, threshold)
    end
    return repartition(_soft_threshold(repartition(tensor, 1, 2), threshold), 2, 1)
end

_update_regularization!(::Nothing, state::SplitState, site::Int) = nothing
function _update_regularization!(buffers::NuclearNormState, state::SplitState, site::Int)
    xi = _penalty(state)
    normalized = state.approximation[site] / _site_scale(state)
    auxiliary = _threshold_pair_bond(normalized - buffers.dual[site] / xi, site, xi)
    buffers.auxiliary[site] = auxiliary
    buffers.dual[site] += xi * (auxiliary - normalized)
    return nothing
end
