_cache_element_name(::PeriodicCache{T}) where {T} = nameof(T)

function _cache_valid_counts(cache::PeriodicCache)
    site_count = length(cache)
    left_count = clamp(cache.left_valid_through + 1, 0, site_count)
    right_count = clamp(site_count - cache.right_valid_from + 1, 0, site_count)
    return left_count, right_count
end

function Base.summary(io::IO, cache::PeriodicCache)
    site_count = length(cache)
    left_count, right_count = _cache_valid_counts(cache)
    print(
        io,
        site_count,
        "-site PeriodicCache{",
        _cache_element_name(cache),
        "} (left ",
        left_count,
        "/",
        site_count,
        ", right ",
        right_count,
        "/",
        site_count,
        " valid)",
    )
    return nothing
end

Base.show(io::IO, cache::PeriodicCache) = summary(io, cache)

_periodic_rung_width(site_count::Int) = ndigits(site_count) + 6

function _periodic_rung_core(site::Int, site_count::Int)
    label = "[" * lpad(site, ndigits(site_count)) * "]"
    return "──" * label * "──"
end

function _periodic_loop_cap(site_count::Int, top::Bool)
    corner_left, corner_right = top ? ("╭─╮", "╭─╮") : ("╰─╯", "╰─╯")
    return corner_left * repeat(" ", _periodic_rung_width(site_count)) *
        corner_right
end

function _periodic_bond_skeleton(site_count::Int)
    return "│ │" * repeat(" ", _periodic_rung_width(site_count)) * "│ │"
end

function _periodic_space_labels(io::IO, space)
    if length(space) == 2
        return (
            sprint(show, space[1]; context=io),
            sprint(show, space[2]; context=io),
        )
    end
    return sprint(show, space; context=io), ""
end

function _periodic_left_space_width(io::IO, spaces)
    return maximum(
        textwidth(first(_periodic_space_labels(io, space)))
        for space in spaces
    )
end

function _periodic_graph_line(line::AbstractString, left_width::Int)
    return repeat(" ", left_width + 1) * line
end

function _periodic_bond_line(
        io::IO,
        site_count::Int,
        space,
        left_width::Int,
    )
    left_label, right_label = _periodic_space_labels(io, space)
    left_padding = left_width - textwidth(left_label)
    line = repeat(" ", left_padding) * left_label * " " *
        _periodic_bond_skeleton(site_count)
    isempty(right_label) || (line *= " " * right_label)
    return line
end

function _cache_node(cache::PeriodicCache, side::Symbol, site::Int)
    site_count = length(cache)
    if (side == :left && site == 1) ||
            (side == :right && site == site_count)
        return "⊙"
    end
    valid = side == :left ?
        site - 1 <= cache.left_valid_through :
        site >= cache.right_valid_from
    return valid ? "●" : "○"
end

function _cache_rung(cache::PeriodicCache, site::Int)
    left_node = _cache_node(cache, :left, site)
    right_node = _cache_node(cache, :right, site)
    return "│ " * left_node * _periodic_rung_core(site, length(cache)) *
        right_node * " │"
end

function _cache_cut_space(cache::PeriodicCache, site::Int)
    site == 1 && return edge_space(cache)
    prefix = cache.left[site]
    return prefix isa FormalIdentity ? prefix.space : domain(prefix)
end

function Base.show(io::IO, ::MIME"text/plain", cache::PeriodicCache)
    summary(io, cache)
    get(io, :compact, false)::Bool && return nothing

    println(io, ":")
    site_count = length(cache)
    limit = get(io, :limit, true)::Bool
    half_screen_sites = limit ?
        max(1, div(displaysize(io)[1] - 4, 4)) : typemax(Int)
    show_all_sites = !limit || site_count <= 2 * half_screen_sites
    bond_spaces = Any[
        edge_space(cache),
        (_cache_cut_space(cache, site) for site in 2:site_count)...,
    ]
    left_width = _periodic_left_space_width(io, bond_spaces)

    println(io, _periodic_graph_line(
        _periodic_loop_cap(site_count, true), left_width
    ))
    println(io, _periodic_bond_line(
        io, site_count, edge_space(cache), left_width
    ))
    for site in reverse(1:site_count)
        show_site = show_all_sites ||
            site > site_count - half_screen_sites || site <= half_screen_sites
        if show_site
            println(io, _periodic_graph_line(
                _cache_rung(cache, site), left_width
            ))
            site == 1 || println(
                io,
                _periodic_bond_line(
                    io,
                    site_count,
                    _cache_cut_space(cache, site),
                    left_width,
                ),
            )
        elseif site == half_screen_sites + 1
            println(
                io,
                _periodic_graph_line(
                    _periodic_bond_skeleton(site_count), left_width
                ),
                " ⋮",
            )
        end
    end
    println(io, _periodic_graph_line(
        _periodic_loop_cap(site_count, false), left_width
    ))
    print(
        io,
        repeat(" ", left_width + 1),
        "⊙ identity   ● valid   ○ stale",
    )
    return nothing
end
