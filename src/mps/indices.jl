# Half-integer bond positions and periodically indexed bond collections.

struct HalfPeriodicVector{A} <: AbstractVector{A}
    elements::PeriodicVector{A}
end

Base.size(v::HalfPeriodicVector) = size(parent(v))
Base.parent(v::HalfPeriodicVector) = v.elements
Base.IndexStyle(::Type{<:HalfPeriodicVector}) = IndexLinear()
function Base.getindex(v::HalfPeriodicVector, position::HalfInt)
    ishalfodd(position) || throw(ArgumentError(
        "expected a half-odd-integer bond position, got $position"
    ))
    return v.elements[floor(Int, position)]
end
Base.getindex(v::HalfPeriodicVector, i::Int) = v.elements[i]
Base.getindex(v::HalfPeriodicVector, x::Rational{Int}) = v[HalfInt(x)]

HalfPeriodicVector(v::AbstractVector) = HalfPeriodicVector(PeriodicVector(v))
