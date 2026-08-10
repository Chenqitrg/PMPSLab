"""Generate each factor once, staging factors `2:N` in right-cache storage."""
function stage_factors(
        N::Int,
        generate_factor::F,
        space_from_factor::G,
    ) where {F,G}
    N > 0 || throw(ArgumentError("cannot cache an empty periodic product"))
    first_factor = generate_factor(1)
    T = typeof(first_factor)
    space = space_from_factor(first_factor)
    S = typeof(space)
    staged_right = Vector{UnitalElement{T,S}}(undef, N)
    staged_right[N] = FormalIdentity{T}(space)
    last_factor = first_factor

    for site in 2:N
        last_factor = generate_factor(site)
        staged_right[site - 1] = last_factor
    end
    compose_product(staged_right[N], first_factor)
    return first_factor, last_factor, staged_right, space
end

"""Build prefixes from the first factor and staged remaining factors."""
function accumulate_prefixes(
        first_factor::T,
        staged_right::Vector{UnitalElement{T,S}},
        space::S,
    ) where {T,S}
    N = length(staged_right)
    left = Vector{UnitalElement{T,S}}(undef, N)
    left[1] = FormalIdentity{T}(space)
    if N > 1
        left[2] = first_factor
    end
    for site in 2:N-1
        left[site + 1] = left[site] * staged_right[site - 1]
    end
    return left
end

"""Transform staged right-cache factors into suffix products in place."""
function accumulate_suffixes!(
        staged_right::Vector{UnitalElement{T,S}},
    ) where {T,S}
    N = length(staged_right)
    for site in N-2:-1:1
        staged_right[site] = staged_right[site] * staged_right[site + 1]
    end
    return staged_right
end

"""
    build_cache(N::Int, generate_factor, space_from_factor)

Incrementally build a cache without materializing an `AbstractPeriodicProduct`.
Each factor is generated exactly once. The right-cache storage stages factors
`2:N` before being transformed in place into suffix products, avoiding factor
copies and additional factor-sized storage.
"""
function build_cache(
        N::Int,
        generate_factor::F,
        space_from_factor::G,
    ) where {F,G}
    first_factor, last_factor, right, space = stage_factors(
        N, generate_factor, space_from_factor
    )
    compose_product(last_factor, right[N])
    left = accumulate_prefixes(first_factor, right, space)
    accumulate_suffixes!(right)
    T = typeof(first_factor)
    return PeriodicCache{T,typeof(space)}(
        left, right, N - 1, 1
    )
end
