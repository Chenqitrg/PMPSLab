"""
    PeriodicCache{T}

O(N) partial-product cache for a periodic product split at `site`.

Stores all left-prefix and right-suffix products of an
[`AbstractPeriodicProduct`](@ref),
so any subrange on either side of the cut can be retrieved without recomputation.

# Cached products (N = length of the periodic product)

    cache[1]  = v[1]
    cache[2]  = v[1] * v[2]
    ...
    cache[site - 1]  = v[1] * ... * v[site - 1]         

    cache[site] = v[site + 1] * ... * v[N]
    cache[site + 1] = v[site + 2] * ... * v[N]
    ...
    cache[N - 1] = v[N]       

# Range queries via `getindex(v, r)`

The full environment at `site` (product of everything except `v[site]`) is:

    environment = right[site] * left[site-1]

# Fields
- `site::Int`: the active cut position (1 ≤ site ≤ N)
- `edge_identity::T`: the identity at the cut, used when half a period is empty
"""
mutable struct PeriodicCache{T}
    cache::Vector{T}
    site::Int
    edge_identity::T
end

Base.length(v::PeriodicCache) = length(v.cache) + 1
Base.lastindex(v::PeriodicCache) = length(v)

"""
    v[r::AbstractUnitRange] -> product

Return the product over a periodic subrange, with `site` as the cut:

- `r[1] == 1` and `r[end] < site`: the prefix `v[1] * ... * v[r[end]]` — cached in `v.cache[r[end]]`.
- `r[1] > site` and `r[end] == length(v)`: the suffix `v[r[1]] * ... * v[N]` — cached in `v.cache[r[1]-1]`.

Any other range is not representable with a single cached product and
throws an [`ArgumentError`](@ref).
"""
function Base.getindex(v::PeriodicCache, r::AbstractUnitRange)
    isempty(r) && throw(ArgumentError("Range must be non-empty"))
    if r[1] == 1 && r[end] < v.site
        return v.cache[r[end]]
    elseif r[1] > v.site && r[end] == length(v)
        return v.cache[r[1] - 1]
    else
        throw(ArgumentError("The range $r is not valid"))
    end
end

"""
    left_cache(v::PeriodicCache)

Product `v[1] * ... * v[site-1]` (the left half of the period, O(1)).
Returns `v.edge_identity` when `site == 1` (empty half).
"""
function left_cache(v::PeriodicCache)
    i = v.site
    i == 1 && return v.edge_identity
    return v[1 : i - 1]
end

"""
    right_cache(v::PeriodicCache)

Product `v[site+1] * ... * v[N]` (the right half of the period, O(1)).
Returns `v.edge_identity` when `site == length(v)` (empty half).
"""
function right_cache(v::PeriodicCache)
    i = v.site
    i == length(v) && return v.edge_identity
    return v[i + 1 : end]
end

"""
    build_cache(v::AbstractPeriodicProduct, site::Int = 1) -> PeriodicCache

Build the O(N) partial-product cache for periodic product `v` cut at `site`.
The identity for empty half-periods is obtained via `edge_identity(v)`.

Cache layout (N = length(v)):

    cache[1]        = v[1]
    cache[2]        = v[1] * v[2]
    ...
    cache[site-1]   = v[1] * ... * v[site-1]

    cache[site]     = v[site+1] * ... * v[N]
    cache[site+1]   = v[site+2] * ... * v[N]
    ...
    cache[N-1]      = v[N]

Filled with two passes (prefix from the left, suffix from the right),
totalling N-2 multiplications for N > 1.
"""
function build_cache(v::AbstractPeriodicProduct, site::Int = 1)
    N = length(v)
    N > 0 || throw(ArgumentError("cannot cache an empty periodic product"))
    1 <= site <= N || throw(BoundsError(v, site))
    T = typeof(v[1])
    cache = Vector{T}(undef, N - 1)

    if N > 1
        acc = v[1]
        for k in 1:site-1
            cache[k] = acc
            k < site-1 && (acc = acc * v[k + 1])
        end

        acc = v[N]
        for k in N-1:-1:site
            cache[k] = acc
            k > site && (acc = v[k] * acc)
        end
    end

    return PeriodicCache(cache, site, edge_identity(v))
end

"""
    environment(v::PeriodicCache)

The product of every element except `v[site]`:

    environment = (v[site+1] * ... * v[N]) * (v[1] * ... * v[site-1])

Computed as `right_cache(v) * left_cache(v)` — two O(1) cache lookups
plus one multiplication. This is the "environment" felt by site `site`,
the input to a local update rule in a sweep.
"""
environment(v::PeriodicCache) = right_cache(v) * left_cache(v)

"""
    advance!(v::PeriodicCache, product, st::SweepState)

Slide the cut of `v` after `st.site` was updated (the fresh element is
already stored in `product[site]`). `st.dir` = +1 advances forward, -1 backward.

Interior moves are O(1): forward, slot `s` becomes `cache[s-1] * product[s]`;
backward, slot `s-1` becomes `product[s] * cache[s]`. Crossing the period
boundary rebuilds the whole cache in O(N) from `product`.
"""
function advance!(v::PeriodicCache, product::AbstractPeriodicProduct, st::SweepState)
    N = length(v)
    s = v.site
    if st.dir == 1 && s < N
        v.cache[s] = s == 1 ? product[s] : v.cache[s-1] * product[s]
        v.site = s + 1
    elseif st.dir == -1 && s > 1
        v.cache[s-1] = s == N ? product[s] : product[s] * v.cache[s]
        v.site = s - 1
    else
        fresh = build_cache(product, st.dir == 1 ? 1 : N)
        copyto!(v.cache, fresh.cache)
        v.site = fresh.site
    end
    return v
end
