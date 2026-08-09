# Half-integer bond positions and periodically indexed bond collections.

struct HalfInt
    n::Int  # represents n + 1/2, i.e. the bond between site n and site n+1
end

Base.convert(::Type{HalfInt}, n::Int) = HalfInt(n)

function HalfInt(x::Rational{Int})
    denominator(x) == 2 || throw(ArgumentError(
        "HalfInt: expected a half-integer n + 1/2 (denominator 2), got $x"
    ))
    # numerator is odd (lowest terms), so n = (num - 1) / 2 exactly;
    # div truncates toward zero, which would give 0 for -1//2.
    return HalfInt(div(numerator(x) - 1, 2))
end

Base.hash(h::HalfInt, h0::UInt) = hash(h.n, h0)
Base.:(==)(a::HalfInt, b::HalfInt) = a.n == b.n
Base.:+(h::HalfInt, n::Int) = HalfInt(h.n + n)
Base.:-(h::HalfInt, n::Int) = HalfInt(h.n - n)

struct HalfPeriodicVector{A} <: AbstractVector{A}
    elements::PeriodicVector{A}
end

Base.size(v::HalfPeriodicVector) = size(parent(v))
Base.parent(v::HalfPeriodicVector) = v.elements
Base.IndexStyle(::Type{<:HalfPeriodicVector}) = IndexLinear()
Base.getindex(v::HalfPeriodicVector, pos::HalfInt) = v.elements[pos.n]
Base.getindex(v::HalfPeriodicVector, i::Int) = v.elements[i]
Base.getindex(v::HalfPeriodicVector, x::Rational{Int}) = v[HalfInt(x)]

HalfPeriodicVector(v::AbstractVector) = HalfPeriodicVector(PeriodicVector(v))
