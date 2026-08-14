function Base.summary(io::IO, transfer::DenseTransferMatrix)
    data = dense(transfer)
    print(
        io,
        dim(codomain(data)),
        "×",
        dim(domain(data)),
        " DenseTransferMatrix(",
        string(scalartype(data)),
        ", ",
        TensorKit.type_repr(spacetype(data)),
        ")",
    )
    return nothing
end

Base.show(io::IO, transfer::DenseTransferMatrix) = summary(io, transfer)

function _dense_transfer_bond_line(
        io::IO,
        space,
        left_width::Int,
        rung_width::Int,
    )
    left_label, right_label = _periodic_space_labels(io, space)
    left_padding = left_width - textwidth(left_label)
    return repeat(" ", left_padding) * left_label * " │" *
        repeat(" ", rung_width) * "│ " * right_label
end

function Base.show(io::IO, ::MIME"text/plain", transfer::DenseTransferMatrix)
    summary(io, transfer)
    get(io, :compact, false)::Bool && return nothing

    data = dense(transfer)
    numout(data) == 2 && numin(data) == 2 || return nothing
    top_space = domain(data)
    bottom_space = codomain(data)
    left_width = _periodic_left_space_width(io, (top_space, bottom_space))
    rung_width = _periodic_rung_width(1)

    println(io, ":")
    println(io, _dense_transfer_bond_line(
        io, top_space, left_width, rung_width
    ))
    println(
        io,
        repeat(" ", left_width + 1),
        "├",
        repeat("─", rung_width),
        "┤",
    )
    print(io, _dense_transfer_bond_line(
        io, bottom_space, left_width, rung_width
    ))
    return nothing
end

_periodic_transfer_element_name(::PeriodicTransferMatrix{T}) where {T} = nameof(T)

function Base.summary(io::IO, transfer::PeriodicTransferMatrix)
    first_transfer = transfer[1]
    print(
        io,
        length(transfer),
        "-site PeriodicTransferMatrix{",
        _periodic_transfer_element_name(transfer),
        "}(",
        string(scalartype(first_transfer)),
        ", ",
        TensorKit.type_repr(spacetype(codomain(first_transfer))),
        ")",
    )
    return nothing
end

Base.show(io::IO, transfer::PeriodicTransferMatrix) = summary(io, transfer)

function _periodic_transfer_rung(
        transfer::PeriodicTransferMatrix,
        site::Int,
    )
    return "│ ├" * _periodic_rung_core(site, length(transfer)) * "┤ │"
end

function Base.show(
        io::IO,
        ::MIME"text/plain",
        transfer::PeriodicTransferMatrix,
    )
    summary(io, transfer)
    get(io, :compact, false)::Bool && return nothing

    println(io, ":")
    site_count = length(transfer)
    limit = get(io, :limit, true)::Bool
    half_screen_sites = limit ?
        max(1, div(displaysize(io)[1] - 4, 4)) : typemax(Int)
    show_all_sites = !limit || site_count <= 2 * half_screen_sites
    bond_spaces = Any[
        domain(transfer[site_count]),
        (codomain(transfer[site]) for site in 2:site_count)...,
    ]
    left_width = _periodic_left_space_width(io, bond_spaces)

    println(io, _periodic_graph_line(
        _periodic_loop_cap(site_count, true), left_width
    ))
    println(io, _periodic_bond_line(
        io, site_count, domain(transfer[site_count]), left_width
    ))
    for site in reverse(1:site_count)
        show_site = show_all_sites ||
            site > site_count - half_screen_sites || site <= half_screen_sites
        if show_site
            println(io, _periodic_graph_line(
                _periodic_transfer_rung(transfer, site), left_width
            ))
            site == 1 || println(
                io,
                _periodic_bond_line(
                    io,
                    site_count,
                    codomain(transfer[site]),
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
    print(io, _periodic_graph_line(
        _periodic_loop_cap(site_count, false), left_width
    ))
    return nothing
end
