"""
    build_cache(N::Int, product_element_at)

Build a [`PeriodicCache`](@ref) directly from the callable

    product_element_at(site::Int) -> product element at site

The returned cache has the site-aligned meanings

    left[site]  = element[1] * ... * element[site - 1]
    right[site] = element[site + 1] * ... * element[N]

`product_element_at` is called twice for every site across the two directional
scans and their boundary checks. Repeated calls at the same site must return
equivalent elements of one concrete type. Adjacent elements must have compatible
domains and codomains, and the final domain must equal the first codomain.

This trades repeated element construction for avoiding a separate period-sized
element collection.
"""
function build_cache(N::Int, product_element_at)
    N > 0 || throw(ArgumentError("cannot cache an empty periodic product"))

    first_element = product_element_at(1)
    T = typeof(first_element)
    space = codomain(first_element)
    S = typeof(space)

    left = Vector{UnitalElement{T,S}}(undef, N)
    right = Vector{UnitalElement{T,S}}(undef, N)
    left[1] = FormalIdentity{T}(space)
    right[N] = FormalIdentity{T}(space)

    # --- Prefixes -------------------------------------------------------------
    for site in 2:N
        left[site] = compose_product(
            left[site - 1], product_element_at(site - 1)
        )
    end

    # --- Suffixes -------------------------------------------------------------
    for site in N-1:-1:1
        right[site] = compose_product(
            product_element_at(site + 1), right[site + 1]
        )
    end

    # The second visit to the final site also validates periodic closure when
    # both scan ranges are empty for a one-site product.
    compose_product(product_element_at(N), right[N])

    return PeriodicCache{T,S}(left, right, N - 1, 1)
end
