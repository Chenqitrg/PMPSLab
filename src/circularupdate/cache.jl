"""
    StaleCacheError

Raised when a query would read a prefix or suffix invalidated by factor updates.
No factor multiplication is performed to repair stale state implicitly.
"""
struct StaleCacheError <: Exception
    side::Symbol
    index::Int
    valid_boundary::Int
end

function Base.showerror(io::IO, err::StaleCacheError)
    if err.side == :left
        print(
            io,
            "left cache entry ", err.index,
            " is stale; valid prefixes end at ", err.valid_boundary,
        )
    else
        print(
            io,
            "right cache entry ", err.index,
            " is stale; valid suffixes start at ", err.valid_boundary,
        )
    end
end

"""
    PeriodicCache{T,S}

Validated O(N) prefix/suffix cache for an [`AbstractPeriodicProduct`](@ref).

    left[s]  = v[1] * ... * v[s - 1]
    right[s] = v[s + 1] * ... * v[N]

`left[1]` and `right[N]` are [`FormalIdentity`](@ref) values carrying only the
periodic cut space. Boundary and one-site queries therefore never construct or
contract with a dense identity.

Element updates invalidate a suffix of `left` and a prefix of `right`. The cache
records the two validity frontiers:

    left[s]  is valid when s - 1 <= left_valid_through
    right[s] is valid when s >= right_valid_from

Queries never repair an invalid entry implicitly, because factor multiplication
may be the dominant cost. A stale query throws [`StaleCacheError`](@ref). Public
[`advance!`](@ref) updates exactly one directional slot when its prerequisite is
valid. A periodic wrap requires an explicit directional rebuild.

The cache stores no sweep cursor or separate factor list. Callers own the
product and notify the cache through [`advance!`](@ref) or [`invalidate!`](@ref).
"""
mutable struct PeriodicCache{T,S}
    left::Vector{UnitalElement{T,S}}
    right::Vector{UnitalElement{T,S}}
    left_valid_through::Int
    right_valid_from::Int
end

function edge_space(v::PeriodicCache{T,S}) where {T,S}
    formal_identity = left_cache_before(v, 1)::FormalIdentity{T,S}
    return formal_identity.space
end

function Base.length(v::PeriodicCache)
    left_length = length(v.left)
    right_length = length(v.right)
    left_length == right_length || throw(DimensionMismatch(
        "left cache has $left_length entries but right cache has $right_length"
    ))
    return left_length
end
Base.lastindex(v::PeriodicCache) = length(v)

"""
    v[r::AbstractUnitRange] -> product

Return a valid prefix `v[1:k]` or suffix `v[k:N]`. Stale entries throw
[`StaleCacheError`](@ref); interior, empty, and full-period ranges throw
[`ArgumentError`](@ref).
"""
function Base.getindex(v::PeriodicCache, r::AbstractUnitRange)
    isempty(r) && throw(ArgumentError("Range must be non-empty"))
    if first(r) == 1 && last(r) < length(v)
        k = last(r)
        return left_cache_before(v, k + 1)
    elseif first(r) > 1 && last(r) == length(v)
        k = first(r) - 1
        return right_cache_after(v, k)
    else
        throw(ArgumentError("The range $r is not valid"))
    end
end

"""
    left_cache_before(v::PeriodicCache, site::Int)

Return the prefix before `site`, or a [`FormalIdentity`](@ref) carrying the
periodic cut space at the left edge. Throw [`StaleCacheError`](@ref) rather than
contracting an invalid prefix implicitly.
"""
function left_cache_before(v::PeriodicCache, site::Int)
    prefix = v.left[site]
    prefix_end = site - 1
    prefix_end <= v.left_valid_through || throw(StaleCacheError(
        :left, prefix_end, v.left_valid_through
    ))
    return prefix
end

"""
    right_cache_after(v::PeriodicCache, site::Int)

Return the suffix after `site`, or a [`FormalIdentity`](@ref) carrying the
periodic cut space at the right edge. Throw [`StaleCacheError`](@ref) rather
than contracting an invalid suffix implicitly.
"""
function right_cache_after(v::PeriodicCache, site::Int)
    suffix = v.right[site]
    site >= v.right_valid_from || throw(StaleCacheError(
        :right, site, v.right_valid_from
    ))
    return suffix
end

function _append_product!(
        destination::T,
        prefix::FormalIdentity{T},
        factor::T,
    ) where {T}
    compose_product(prefix, factor)
    return copy!(destination, factor)
end

_append_product!(destination::T, prefix::T, factor::T) where {T} =
    mul!(destination, prefix, factor)

function _prepend_product!(
        destination::T,
        factor::T,
        suffix::FormalIdentity{T},
    ) where {T}
    compose_product(factor, suffix)
    return copy!(destination, factor)
end

_prepend_product!(destination::T, factor::T, suffix::T) where {T} =
    mul!(destination, factor, suffix)

"""
    build_cache(v::AbstractPeriodicProduct{T}) -> PeriodicCache{T,S}

Build all site-aligned prefix and suffix segments. Construction uses
`max(0, 2N - 4)` multiplications and stores `2N - 2` concrete factors plus two
formal identities. Endpoint factors are copied so later in-place refresh
does not alias product factors.
"""
function build_cache(v::AbstractPeriodicProduct{T}) where {T}
    N = length(v)
    N > 0 || throw(ArgumentError("cannot cache an empty periodic product"))
    space = edge_space(v)
    left = prefix_products(v)
    right = suffix_products(v)

    return PeriodicCache{T,typeof(space)}(
        left, right, N - 1, 1
    )
end

"""
    environment(v::PeriodicCache, site::Int)

Return the product of every factor except `site`, in periodic order. The method
never repairs stale state: invalid halves throw [`StaleCacheError`](@ref).
"""
function environment(v::PeriodicCache, site::Int)
    left = left_cache_before(v, site)
    right = right_cache_after(v, site)
    return compose_product(right, left)
end
