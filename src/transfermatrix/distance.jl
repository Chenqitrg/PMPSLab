function _nonnegative_squared(value::Real, scale::Real, quantity::AbstractString)
    value >= 0 && return value
    roundoff = 100 * eps(float(one(value))) * max(abs(scale), one(scale))
    value >= -roundoff && return zero(value)
    throw(DomainError(value, "$quantity is negative beyond roundoff tolerance"))
end

function squared_norm(mps::PeriodicMPS)
    value = real(dot(mps, mps))
    return _nonnegative_squared(value, value, "squared MPS norm")
end
function TensorKit.norm(mps::PeriodicMPS)
    return sqrt(squared_norm(mps))
end

function _squared_distance_data(
        mps1::PeriodicMPS,
        mps2::PeriodicMPS
    )
    n1 = squared_norm(mps1)
    n2 = squared_norm(mps2)
    distance2 = n1 + n2 - 2 * real(dot(mps1, mps2))
    scale = max(abs(n1), abs(n2))
    return _nonnegative_squared(distance2, scale, "squared MPS distance"), n1, n2
end

squared_distance(mps1::PeriodicMPS, mps2::PeriodicMPS) =
    first(_squared_distance_data(mps1, mps2))

# Compatibility name retained for callers that already treated `diff_norm` as
# a squared distance.
diff_norm(mps1::PeriodicMPS, mps2::PeriodicMPS) = squared_distance(mps1, mps2)

function Base.isapprox(
        mps1::PeriodicMPS,
        mps2::PeriodicMPS;
        atol::Real = 0,
        rtol::Real = Base.rtoldefault(scalartype(mps1), scalartype(mps2), atol)
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

function LinearAlgebra.normalize(mps::PeriodicMPS)
    n = norm(mps)
    iszero(n) && throw(ArgumentError("cannot normalize a zero periodic MPS"))
    result = copy(mps)
    result[1] = result[1] / n
    return result
end
