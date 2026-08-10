# --- Prefix scan -------------------------------------------------------------

"""
    prefix_products(v::AbstractPeriodicProduct{T})

Build the site-aligned prefixes where entry `site` represents factors
`1:site-1`. The first entry is a typed formal identity.
"""
function prefix_products(v::AbstractPeriodicProduct{T}) where {T}
    N = length(v)
    N > 0 || throw(ArgumentError(
        "cannot scan an empty periodic product"
    ))

    # --- Boundary identity ---------------------------------------------------
    identity = edge_identity(v)
    prefixes = Vector{UnitalElement{T,typeof(identity.space)}}(undef, N)
    prefixes[1] = identity

    # --- Detached endpoint ---------------------------------------------------
    N > 1 && (prefixes[2] = copy(v[1]))

    # --- Interior scan -------------------------------------------------------
    for site in 2:N-1
        prefixes[site + 1] = (prefixes[site]::T) * v[site]
    end
    return prefixes
end

# --- Suffix scan -------------------------------------------------------------

"""
    suffix_products(v::AbstractPeriodicProduct{T})

Build the site-aligned suffixes where entry `site` represents factors
`site+1:N`. The final entry is a typed formal identity.
"""
function suffix_products(v::AbstractPeriodicProduct{T}) where {T}
    N = length(v)
    N > 0 || throw(ArgumentError(
        "cannot scan an empty periodic product"
    ))

    # --- Boundary identity ---------------------------------------------------
    identity = edge_identity(v)
    suffixes = Vector{UnitalElement{T,typeof(identity.space)}}(undef, N)
    suffixes[N] = identity

    # --- Detached endpoint ---------------------------------------------------
    N > 1 && (suffixes[N - 1] = copy(v[N]))

    # --- Interior scan -------------------------------------------------------
    for site in N-1:-1:2
        suffixes[site - 1] = v[site] * (suffixes[site]::T)
    end
    return suffixes
end
