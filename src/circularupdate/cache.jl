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
"""
mutable struct RingCache{T}
    cache::Vector{T}
    site::Int
end

Base.length(v::RingCache) = length(v.cache) + 1
Base.lastindex(v::RingCache) = length(v)
scalartype(v::RingCache) = scalartype(v.cache[1])

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
    left_right_space(v::RingCache)

The space on which the identity morphism used as unit lives.

Since a `LoopMonoid` is a 1-category (not a plain monoid), the unit
must live in the correct hom-space. For `site == 1` the empty prefix
is `v[1] : A → B`'s codomain-side neighbor, resolved from `cache[end]`
(the last suffix product); otherwise the unit sits at the codomain of
`cache[1]` (the first prefix product).
"""
function left_right_space(v::RingCache)
    if v.site == 1
        return domain(v.cache[end])
    else
        return codomain(v.cache[1])
    end
end

"""
    left_right_unit(v::RingCache)

Identity morphism on [`left_right_space`](@ref) — the unit of the
(1-)category, used when the left or right half of the loop is empty.
"""
function left_right_unit(v::RingCache)
    return id(scalartype(v), left_right_space(v))
end

"""
    left_cache(v::RingCache)

Product `v[1] * ... * v[site-1]` (the "left half" of the loop, O(1)).
Returns the category unit `id` when `site == 1` (empty half).
"""
function left_cache(v::RingCache)
    i = v.site
    i == 1 && return left_right_unit(v)
    return v[1 : i - 1]
end

"""
    right_cache(v::RingCache)

Product `v[site+1] * ... * v[N]` (the "right half" of the loop, O(1)).
Returns the category unit `id` when `site == length(v)` (empty half).
"""
function right_cache(v::RingCache)
    i = v.site
    i == length(v) && return left_right_unit(v)
    return v[i + 1 : end]
end

"""
    build_cache(v::LoopMonoid, site::Int = 1) -> RingCache

Build the O(N) partial-product cache for loop `v` cut at `site`.

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
function build_cache(v::LoopMonoid, site::Int = 1)
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

    return RingCache(cache, site)
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
    advance!(v::RingCache, loop, new_value, dir::Int)

Slide the cut of `v` after `site` was updated to `new_value` (the fresh
element already stored in `loop`). `dir` = +1 slides forward, -1 backward.

Interior moves are O(1): forward, slot `s` becomes `cache[s-1] * new_value`;
backward, slot `s-1` becomes `new_value * cache[s]`. Wrapping around
(N → 1 or 1 → N) rebuilds the whole cache in O(N) from `loop`.
"""
function advance!(v::RingCache, loop::LoopMonoid, new_value, dir::Int)
    N = length(v)
    s = v.site
    if dir == 1 && s < N
        v.cache[s] = s == 1 ? new_value : v.cache[s-1] * new_value
        v.site = s + 1
    elseif dir == -1 && s > 1
        v.cache[s-1] = s == N ? new_value : new_value * v.cache[s]
        v.site = s - 1
    else
        fresh = build_cache(loop, dir == 1 ? 1 : N)
        copyto!(v.cache, fresh.cache)
        v.site = fresh.site
    end
    return v
end