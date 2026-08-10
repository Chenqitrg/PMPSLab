# ------------------------------------------------------------------------------
# AbstractPeriodicProduct — ordered products with periodic indexing
# ------------------------------------------------------------------------------

"""
    AbstractPeriodicProduct{T}

Ordered periodic collection whose factors have type `T`.

Subtypes implement `length`, integer `getindex`, and `edge_identity`. Factor
type `T` is closed under `*` and supports `copy`, `copy!`, and `mul!` for cache
construction and refresh. The protocol is independent of any application-level
factor representation.
"""
abstract type AbstractPeriodicProduct{T} end

Base.eltype(::Type{<:AbstractPeriodicProduct{T}}) where {T} = T

range_product(v::AbstractPeriodicProduct, r::AbstractUnitRange) =
    mapfoldl(i -> v[i], *, r)

# ------------------------------------------------------------------------------
# SweepState — one typed step's position in a sweep
#
#   site — the position to update
#   D    — the scan direction, encoded in the state type
#   n    — completed period count (for stop-check against maximal_sweep)
# ------------------------------------------------------------------------------

abstract type SweepDirection end
struct Forward <: SweepDirection end
struct Backward <: SweepDirection end

struct SweepState{D<:SweepDirection}
    site::Int
    n::Int
end
