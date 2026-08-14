"""
    invalidate!(v::PeriodicCache, site::Int) -> PeriodicCache

Record that element `site` changed without performing a product operation. Every
dependent prefix and suffix becomes unreadable until a legal `advance!` refresh
or an explicit complete-product `rebuild!` makes it valid again.
"""
function invalidate!(v::PeriodicCache, site::Int)
    checkbounds(v.left, site)
    checkbounds(v.right, site)
    v.left_valid_through = min(v.left_valid_through, site - 1)
    v.right_valid_from = max(v.right_valid_from, site)
    return v
end

"""
    rebuild_left!(v::PeriodicCache{T}, product::AbstractPeriodicProduct{T})
    rebuild_left!(v::PeriodicCache{T}, product_element_at)

Explicitly rebuild every non-full prefix from a complete current product.
The callable overload generates each required product element exactly once and
does not materialize an `AbstractPeriodicProduct`.
"""
function rebuild_left!(
        v::PeriodicCache{T},
        product::AbstractPeriodicProduct{T},
    ) where {T}
    _validate_product(v, product)
    return rebuild_left!(v, site -> product[site])
end

function rebuild_left!(
        v::PeriodicCache{T},
        product_element_at,
    ) where {T}
    N = length(v)
    v.left_valid_through = 0
    for site in 1:N-1
        slot = site + 1
        prefix = left_cache_before(v, site)
        v.left[slot] = _append_product!(
            v.left[slot], prefix, product_element_at(site)
        )
        v.left_valid_through = site
    end
    return v
end

"""
    rebuild_right!(v::PeriodicCache{T}, product::AbstractPeriodicProduct{T})
    rebuild_right!(v::PeriodicCache{T}, product_element_at)

Explicitly rebuild every non-full suffix from a complete current product.
The callable overload generates each required product element exactly once and
does not materialize an `AbstractPeriodicProduct`.
"""
function rebuild_right!(
        v::PeriodicCache{T},
        product::AbstractPeriodicProduct{T},
    ) where {T}
    _validate_product(v, product)
    return rebuild_right!(v, site -> product[site])
end

function rebuild_right!(
        v::PeriodicCache{T},
        product_element_at,
    ) where {T}
    N = length(v)
    v.right_valid_from = N
    for site in N:-1:2
        slot = site - 1
        suffix = right_cache_after(v, site)
        v.right[slot] = _prepend_product!(
            v.right[slot], product_element_at(site), suffix
        )
        v.right_valid_from = slot
    end
    return v
end

function _validate_product(
        v::PeriodicCache{T},
        product::AbstractPeriodicProduct{T},
    ) where {T}
    length(product) == length(v) || throw(DimensionMismatch(
        "cache has $(length(v)) sites but product has $(length(product))"
    ))
    product_space = edge_space(product)
    cache_space = edge_space(v)
    product_space == cache_space || throw(SpaceMismatch(
        "product edge space $product_space does not match cache edge space $cache_space"
    ))
    return product
end

_requires_rebuild(v::PeriodicCache, st::SweepState{Forward}) =
    length(v) > 1 && st.site == length(v)

_requires_rebuild(v::PeriodicCache, st::SweepState{Backward}) =
    length(v) > 1 && st.site == 1

function _refresh_prefix!(
        v::PeriodicCache{T}, updated_element::T, site::Int
    ) where {T}
    N = length(v)
    prefix = left_cache_before(v, site)
    for slot in site + 1:min(site + 1, N)
        v.left[slot] = _append_product!(
            v.left[slot], prefix, updated_element
        )
    end
    v.left_valid_through = min(site, N - 1)
    return v
end

function _refresh_suffix!(
        v::PeriodicCache{T}, updated_element::T, site::Int
    ) where {T}
    suffix = right_cache_after(v, site)
    for slot in max(1, site - 1):site - 1
        v.right[slot] = _prepend_product!(
            v.right[slot], updated_element, suffix
        )
    end
    v.right_valid_from = max(1, site - 1)
    return v
end

"""
    advance!(v::PeriodicCache{T}, updated_element::T, st::SweepState{D})

Notify the cache that element `st.site` changed to `updated_element`, then refresh the
prefix or suffix selected by direction type `D`. The method performs at most
one `copy!` or `mul!`.

The validity frontiers are updated before every product operation. If the needed
prerequisite is stale or the transition crosses a periodic boundary, the method
throws and leaves the cache conservatively marked stale. Boundary rebuilds are
explicitly performed with [`rebuild_left!`](@ref) or [`rebuild_right!`](@ref).
"""
function advance!(
        v::PeriodicCache{T},
        updated_element::T,
        st::SweepState{Forward},
    ) where {T}
    s = st.site
    invalidate!(v, s)
    _requires_rebuild(v, st) && throw(ArgumentError(
        "periodic boundary requires an explicit directional rebuild"
    ))
    return _refresh_prefix!(v, updated_element, s)
end

function advance!(
        v::PeriodicCache{T},
        updated_element::T,
        st::SweepState{Backward},
    ) where {T}
    s = st.site
    invalidate!(v, s)
    _requires_rebuild(v, st) && throw(ArgumentError(
        "periodic boundary requires an explicit directional rebuild"
    ))
    return _refresh_suffix!(v, updated_element, s)
end

"""
    rebuild!(v::PeriodicCache{T}, product::AbstractPeriodicProduct{T})

Explicitly rebuild both cache directions from a complete current product.
Unlike a query, this operation is allowed to perform the required contractions.
"""
function rebuild!(
        v::PeriodicCache{T},
        product::AbstractPeriodicProduct{T},
    ) where {T}
    _validate_product(v, product)
    rebuild_left!(v, product)
    rebuild_right!(v, product)
    return v
end
