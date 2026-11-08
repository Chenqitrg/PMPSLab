function _nonnegative_squared(value::Real, scale::Real, quantity::AbstractString)
    value >= 0 && return value
    roundoff = 100 * eps(float(one(value))) * max(abs(scale), one(scale))
    value >= -roundoff && return zero(value)
    throw(DomainError(value, "$quantity is negative beyond roundoff tolerance"))
end

function _scaled_self_transfer(mps::PeriodicMPS, site::Int)
    tensor = mps[site]
    tensor_scale = LinearAlgebra.norm(tensor)
    iszero(tensor_scale) && return nothing
    isfinite(tensor_scale) || throw(DomainError(
        tensor_scale, "cannot normalize an MPS with non-finite tensor norms"
    ))
    transfer = DenseTransferMatrix(tensor / tensor_scale)
    return transfer, 2 * log(tensor_scale)
end

function _rescale_transfer(transfer::DenseTransferMatrix)
    transfer_scale = LinearAlgebra.norm(TensorKit.TensorMap(transfer))
    iszero(transfer_scale) && return nothing
    isfinite(transfer_scale) || throw(DomainError(
        transfer_scale, "non-finite intermediate while normalizing an MPS"
    ))
    scaled = DenseTransferMatrix(TensorKit.TensorMap(transfer) / transfer_scale)
    return scaled, log(transfer_scale)
end

function _log_squared_norm(mps::PeriodicMPS)
    first_transfer = _scaled_self_transfer(mps, 1)
    isnothing(first_transfer) && return -Inf
    product, log_scale = first_transfer

    scaled_product = _rescale_transfer(product)
    isnothing(scaled_product) && return -Inf
    product, product_log_scale = scaled_product
    log_scale += product_log_scale

    for site in 2:length(mps)
        site_transfer = _scaled_self_transfer(mps, site)
        isnothing(site_transfer) && return -Inf
        transfer, transfer_log_scale = site_transfer

        scaled_product = _rescale_transfer(product * transfer)
        isnothing(scaled_product) && return -Inf
        product, product_log_scale = scaled_product
        log_scale += transfer_log_scale + product_log_scale
    end

    trace_value = real(LinearAlgebra.tr(TensorKit.TensorMap(product)))
    trace_value = _nonnegative_squared(
        trace_value, LinearAlgebra.norm(TensorKit.TensorMap(product)), "scaled squared MPS norm"
    )
    iszero(trace_value) && return -Inf
    return log_scale + log(trace_value)
end

function squared_norm(mps::PeriodicMPS)
    return exp(_log_squared_norm(mps))
end
function TensorKit.norm(mps::PeriodicMPS)
    return exp(_log_squared_norm(mps) / 2)
end

function _squared_distance_data(
        mps1::PeriodicMPS,
        mps2::PeriodicMPS
    )
    n1 = squared_norm(mps1)
    n2 = squared_norm(mps2)
    distance2 = n1 + n2 - 2 * real(LinearAlgebra.dot(mps1, mps2))
    scale = max(abs(n1), abs(n2))
    return _nonnegative_squared(distance2, scale, "squared MPS distance"), n1, n2
end

squared_distance(mps1::PeriodicMPS, mps2::PeriodicMPS) =
    first(_squared_distance_data(mps1, mps2))

# Compatibility wrapper for the former name, which misleadingly suggested a
# norm despite returning a squared distance.
function diff_norm(mps1::PeriodicMPS, mps2::PeriodicMPS)
    Base.depwarn("diff_norm returns a squared distance; use squared_distance", :diff_norm)
    return squared_distance(mps1, mps2)
end

function Base.isapprox(
        mps1::PeriodicMPS,
        mps2::PeriodicMPS;
        atol::Real = 0,
        rtol::Real = Base.rtoldefault(TensorKit.scalartype(mps1), TensorKit.scalartype(mps2), atol)
    )
    distance2, n1, n2 = _squared_distance_data(mps1, mps2)
    scale2 = max(abs(n1), abs(n2))
    tolerance = max(atol, rtol * sqrt(scale2))

    # `distance2` subtracts three independently contracted quantities of size
    # `scale2`. At the default `rtol`, squaring the requested tolerance reaches
    # machine precision, so contraction roundoff can otherwise decide the
    # result solely by its sign. Use the same roundoff budget in both
    # directions; unlike `_nonnegative_squared`, keep the reported distance
    # itself unchanged when it is small and positive.
    roundoff2 = 100 * eps(float(one(distance2))) * scale2
    return distance2 <= max(tolerance^2, roundoff2)
end

"""
    normalize(mps::PeriodicMPS)

Return a normalized copy of `mps`, leaving the original state unchanged. The
global normalization is computed with logarithmic scale accumulation and
distributed evenly over all site tensors.
"""
function LinearAlgebra.normalize(mps::PeriodicMPS)
    inverse_site_scale = _inverse_normalization_site_scale(mps)
    return map_tensors(tensor -> tensor * inverse_site_scale, mps)
end

"""
    normalize!(mps::PeriodicMPS)

Normalize `mps` in place and return it. The underlying tensor data is scaled
directly, preserving both the MPS and site-tensor object identities.
"""
function LinearAlgebra.normalize!(mps::PeriodicMPS)
    inverse_site_scale = _inverse_normalization_site_scale(mps)
    foreach(tensor -> LinearAlgebra.rmul!(tensor, inverse_site_scale), mps)
    return mps
end

function _inverse_normalization_site_scale(mps::PeriodicMPS)
    log_squared_norm = _log_squared_norm(mps)
    log_squared_norm == -Inf && throw(ArgumentError(
        "cannot normalize a zero periodic MPS"
    ))
    isfinite(log_squared_norm) || throw(DomainError(
        log_squared_norm, "cannot normalize an MPS with a non-finite norm"
    ))

    site_scale = exp(log_squared_norm / (2 * length(mps)))
    return inv(site_scale)
end
