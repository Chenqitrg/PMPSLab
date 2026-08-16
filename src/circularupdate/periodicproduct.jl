# ------------------------------------------------------------------------------
# AbstractPeriodicProduct — ordered products with periodic indexing
# ------------------------------------------------------------------------------

"""
    AbstractPeriodicProduct{T}

Ordered periodic collection whose factors have type `T`.

Subtypes implement `length` and integer `getindex`. Factors expose `domain` and
`codomain`; the final domain must equal the first codomain.
`T` is closed under `*` and supports `copy`, `copy!`, and `mul!` for cache
construction and refresh. The protocol is independent of any application-level
factor representation.
"""
abstract type AbstractPeriodicProduct{T} end

Base.eltype(::Type{<:AbstractPeriodicProduct{T}}) where {T} = T

function edge_space(v::AbstractPeriodicProduct)
    N = length(v)
    N > 0 || throw(ArgumentError(
        "cannot determine the edge space of an empty periodic product"
    ))
    first_codomain = codomain(v[1])
    last_domain = domain(v[N])
    last_domain == first_codomain || throw(SpaceMismatch(
        "final domain $last_domain does not match first codomain $first_codomain"
    ))
    return first_codomain
end

range_product(v::AbstractPeriodicProduct, r::AbstractUnitRange) =
    mapfoldl(i -> v[i], *, r)

# ------------------------------------------------------------------------------
# SweepState — one typed step's position in a sweep
#
#   site — the position to update
#   D    — the scan direction, encoded in the state type
#   completed_sweeps — completed period count
# ------------------------------------------------------------------------------

abstract type SweepDirection end
struct Forward <: SweepDirection end
struct Backward <: SweepDirection end

struct SweepState{D<:SweepDirection}
    site::Int
    completed_sweeps::Int
end
