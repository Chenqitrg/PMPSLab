"""
    RingCache{T}

O(N) partial-product cache for a periodic loop split at `site`.

Stores all left-prefix and right-suffix products of a [`LoopMonoid`](@ref),
so any subrange on either side of the cut can be retrieved without recomputation.

# Cached products (N = length of the loop)

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
- `unit::T`: the category identity, used when a half-loop is empty
"""
mutable struct RingCache{T}
    cache::Vector{T}
    site::Int
    unit::T
end

Base.length(v::RingCache) = length(v.cache) + 1
Base.lastindex(v::RingCache) = length(v)

"""
    v[r::AbstractUnitRange] -> product

Return the product over a subrange of the loop, with `site` as the cut:

- `r[1] == 1` and `r[end] < site`: the prefix `v[1] * ... * v[r[end]]` — cached in `v.cache[r[end]]`.
- `r[1] > site` and `r[end] == length(v)`: the suffix `v[r[1]] * ... * v[N]` — cached in `v.cache[r[1]-1]`.

Any other range is not representable with a single cached product and
throws an [`ArgumentError`](@ref).
"""
function Base.getindex(v::RingCache, r::AbstractUnitRange)
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
    left_cache(v::RingCache)

Product `v[1] * ... * v[site-1]` (the "left half" of the loop, O(1)).
Returns `v.unit` when `site == 1` (empty half).
"""
function left_cache(v::RingCache)
    i = v.site
    i == 1 && return v.unit
    return v[1 : i - 1]
end

"""
    right_cache(v::RingCache)

Product `v[site+1] * ... * v[N]` (the "right half" of the loop, O(1)).
Returns `v.unit` when `site == length(v)` (empty half).
"""
function right_cache(v::RingCache)
    i = v.site
    i == length(v) && return v.unit
    return v[i + 1 : end]
end

"""
    build_cache(v::LoopMonoid, site::Int, unit) -> RingCache

Build the O(N) partial-product cache for loop `v` cut at `site`.
`unit` is the category identity for empty half-loops (provided by caller).

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
function build_cache(v::LoopMonoid, site::Int, unit)
    N = length(v)
    T = eltype(v)
    cache = Vector{T}(undef, N - 1)

    # left[k] = v[1] * ... * v[k]
    if N > 1
        acc = v[1]
        for k in 1:site-1
            cache[k] = acc
            k < site-1 && (acc = acc * v[k + 1])
        end

        # right[k] = v[k+1] * ... * v[N]
        acc = v[N]
        for k in N-1:-1:site
            cache[k] = acc
            k > site && (acc = v[k] * acc)
        end
    end

    return RingCache(cache, site, unit)
end

"""
    environment(v::RingCache)

The product of every element except `v[site]`:

    environment = (v[site+1] * ... * v[N]) * (v[1] * ... * v[site-1])

Computed as `right_cache(v) * left_cache(v)` — two O(1) cache lookups
plus one multiplication. This is the "environment" felt by site `site`,
the input to a local update rule in a sweep.
"""
environment(v::RingCache) = right_cache(v) * left_cache(v)

"""
    advance!(v::RingCache, loop, st::SweepState)

Slide the cut of `v` after `st.site` was updated (the fresh element is
already stored in `loop[site]`). `st.dir` = +1 slides forward, -1 backward.

Interior moves are O(1): forward, slot `s` becomes `cache[s-1] * loop[s]`;
backward, slot `s-1` becomes `loop[s] * cache[s]`. Wrapping around
(N → 1 or 1 → N) rebuilds the whole cache in O(N) from `loop`.
"""
function advance!(v::RingCache, loop::LoopMonoid, st::SweepState)
    N = length(v)
    s = v.site
    if st.dir == 1 && s < N
        v.cache[s] = s == 1 ? loop[s] : v.cache[s-1] * loop[s]
        v.site = s + 1
    elseif st.dir == -1 && s > 1
        v.cache[s-1] = s == N ? loop[s] : loop[s] * v.cache[s]
        v.site = s - 1
    else
        fresh = build_cache(loop, st.dir == 1 ? 1 : N, v.unit)
        copyto!(v.cache, fresh.cache)
        v.site = fresh.site
    end
    return v
end