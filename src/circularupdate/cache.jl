"""
    PeriodicCache{T}

O(N) partial-product cache for an [`AbstractPeriodicProduct`](@ref).

Left prefixes and right suffixes live in separate arrays with fixed meanings:

    left[k]  = v[1] * ... * v[k]
    right[k] = v[k + 1] * ... * v[N]

The cache deliberately stores no active site. A caller supplies the site to
[`environment`](@ref), so the sweep iterator remains the single source of
position. Keeping each slot's meaning fixed also lets [`advance!`](@ref) update
its tensor storage in place.

# Fields
- `left::Vector{T}`: all non-full left prefixes
- `right::Vector{T}`: all non-full right suffixes
- `edge_identity::T`: the identity used when one side is empty
"""
struct PeriodicCache{T}
    left::Vector{T}
    right::Vector{T}
    edge_identity::T
end

Base.length(v::PeriodicCache) = length(v.left) + 1
Base.lastindex(v::PeriodicCache) = length(v)

function _check_site(v::PeriodicCache, site::Int)
    1 <= site <= length(v) || throw(BoundsError(v, site))
    return site
end

"""
    v[r::AbstractUnitRange] -> product

Return a cached prefix `v[1:k]` or suffix `v[k:N]`. Interior ranges and the
full period are not represented by a single cache entry and throw an
[`ArgumentError`](@ref).
"""
function Base.getindex(v::PeriodicCache, r::AbstractUnitRange)
    isempty(r) && throw(ArgumentError("Range must be non-empty"))
    if first(r) == 1 && last(r) < length(v)
        return v.left[last(r)]
    elseif first(r) > 1 && last(r) == length(v)
        return v.right[first(r) - 1]
    else
        throw(ArgumentError("The range $r is not valid"))
    end
end

"""
    left_cache(v::PeriodicCache, site::Int)

Return `v[1] * ... * v[site-1]` in O(1), or `v.edge_identity` when
`site == 1`.
"""
function left_cache(v::PeriodicCache, site::Int)
    _check_site(v, site)
    site == 1 && return v.edge_identity
    return v.left[site - 1]
end

"""
    right_cache(v::PeriodicCache, site::Int)

Return `v[site+1] * ... * v[N]` in O(1), or `v.edge_identity` when
`site == length(v)`.
"""
function right_cache(v::PeriodicCache, site::Int)
    _check_site(v, site)
    site == length(v) && return v.edge_identity
    return v.right[site]
end

"""
    build_cache(v::AbstractPeriodicProduct) -> PeriodicCache

Build every non-full left prefix and right suffix of `v`. Construction uses
`2N - 4` multiplications for `N > 1` and stores `2N - 2` tensors. Endpoint
tensors are copied so later in-place cache updates never alias `v` itself.
Product elements must support `copy`, `copy!`, and three-argument `mul!` while
preserving the tensor space associated with each site.
"""
function build_cache(v::AbstractPeriodicProduct)
    N = length(v)
    N > 0 || throw(ArgumentError("cannot cache an empty periodic product"))
    T = typeof(v[1])
    left = Vector{T}(undef, N - 1)
    right = Vector{T}(undef, N - 1)

    if N > 1
        left[1] = copy(v[1])
        for k in 2:N-1
            left[k] = left[k - 1] * v[k]
        end

        right[N - 1] = copy(v[N])
        for k in N-2:-1:1
            right[k] = v[k + 1] * right[k + 1]
        end
    end

    return PeriodicCache(left, right, edge_identity(v))
end

"""
    environment(v::PeriodicCache, site::Int)

Return the product of every element except `v[site]`, in periodic order:

    v[site+1] * ... * v[N] * v[1] * ... * v[site-1]

The two partial products are O(1) lookups; an interior site needs one
multiplication. Boundary sites return an existing cached half directly.
"""
function environment(v::PeriodicCache, site::Int)
    _check_site(v, site)
    length(v) == 1 && return v.edge_identity
    site == 1 && return v.right[1]
    site == length(v) && return v.left[end]
    return v.right[site] * v.left[site - 1]
end

function _rebuild_left!(v::PeriodicCache, product::AbstractPeriodicProduct)
    N = length(v)
    N == 1 && return v
    copy!(v.left[1], product[1])
    for k in 2:N-1
        mul!(v.left[k], v.left[k - 1], product[k])
    end
    return v
end

function _rebuild_right!(v::PeriodicCache, product::AbstractPeriodicProduct)
    N = length(v)
    N == 1 && return v
    copy!(v.right[N - 1], product[N])
    for k in N-2:-1:1
        mul!(v.right[k], product[k + 1], v.right[k + 1])
    end
    return v
end

function _validate_advance(v::PeriodicCache, st::SweepState)
    N = length(v)
    s = _check_site(v, st.site)
    st.dir in (-1, 1) || throw(ArgumentError(
        "sweep direction must be -1 or 1, got $(st.dir)"
    ))
    return N, s
end

"""
    _requires_rebuild(v::PeriodicCache, st::SweepState) -> Bool

Return whether advancing at `st` crosses a periodic boundary and therefore
requires the opposite side of the cache to be rebuilt from a complete product.
"""
function _requires_rebuild(v::PeriodicCache, st::SweepState)
    N, s = _validate_advance(v, st)
    return N > 1 && ((st.dir == 1 && s == N) || (st.dir == -1 && s == 1))
end

"""
    advance!(v::PeriodicCache, transfer, st::SweepState)

Update the directional cache after the factor at `st.site` changes. `transfer`
is the new factor itself, not a complete periodic product. This method handles
ordinary movement and zigzag turns without retaining all source factors. A
periodic boundary must instead use the complete-product overload.
"""
function advance!(v::PeriodicCache, transfer, st::SweepState)
    N, s = _validate_advance(v, st)
    N == 1 && return v
    _requires_rebuild(v, st) && throw(ArgumentError(
        "advancing across a periodic boundary requires a complete product"
    ))

    if st.dir == 1
        if s == 1
            copy!(v.left[1], transfer)
        else
            mul!(v.left[s], v.left[s - 1], transfer)
        end
    else
        if s == N
            copy!(v.right[N - 1], transfer)
        else
            mul!(v.right[s - 1], transfer, v.right[s])
        end
    end
    return v
end

"""
    advance!(v::PeriodicCache, product::AbstractPeriodicProduct, st::SweepState)

Update a cache from a complete product. At ordinary sites only the changed
factor is read. Crossing a periodic boundary rebuilds the opposite cache side.
"""
function advance!(
        v::PeriodicCache,
        product::AbstractPeriodicProduct,
        st::SweepState,
    )
    length(product) == length(v) || throw(DimensionMismatch(
        "cache has $(length(v)) sites but product has $(length(product))"
    ))
    _, s = _validate_advance(v, st)
    if _requires_rebuild(v, st)
        if st.dir == 1
            _rebuild_right!(v, product)
        else
            _rebuild_left!(v, product)
        end
        return v
    end
    return advance!(v, product[s], st)
end
