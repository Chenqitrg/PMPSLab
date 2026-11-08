function Base.summary(io::IO, mps::PeriodicMPS)
    maximal_bond_dimension = maximum(
        dim(left_virtualspace(mps, site)) for site in eachsite(mps)
    )
    element_type = scalartype(mps)
    space_type = TensorKit.type_repr(spacetype(mps))
    print(
        io,
        length(mps),
        "-site PeriodicMPS(",
        element_type,
        ", ",
        space_type,
        ") with maximal bond dimension ",
        maximal_bond_dimension,
    )
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", mps::PeriodicMPS)
    summary(io, mps)
    get(io, :compact, false)::Bool && return nothing

    println(io, ":")
    io = IOContext(io, :typeinfo => spacetype(mps))

    limit = get(io, :limit, true)::Bool
    half_screen_sites = limit ? max(1, div(displaysize(io)[1] - 4, 4)) : typemax(Int)
    site_count = length(mps)
    show_all_sites = !limit || site_count <= 2 * half_screen_sites

    println(io, "╭─╮")
    println(io, "│ │ ", right_virtualspace(mps, site_count))
    for site in reverse(eachsite(mps))
        show_site = show_all_sites ||
            site > site_count - half_screen_sites || site <= half_screen_sites
        if show_site
            println(io, "│ ├─[$site]─ ", physicalspace(mps, site))
            site == 1 || println(io, "│ │ ", left_virtualspace(mps, site))
        elseif site == half_screen_sites + 1
            println(io, "│ │ ⋮")
        end
    end
    println(io, "╰─╯")

    return nothing
end
