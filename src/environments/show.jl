# Periodic transfer display is self-contained. Numerical code does not import
# rendering helpers, and this file does not reach into another package's internals.
function Base.summary(io::IO, transfer::PeriodicTransferMatrix{T}) where {T}
    first_transfer = transfer[1]
    print(io, length(transfer), "-site PeriodicTransferMatrix{", nameof(T), "}(",
        string(TensorKit.scalartype(first_transfer)), ", ",
        TensorKit.type_repr(TensorKit.spacetype(TensorKit.codomain(first_transfer))), ")")
    return nothing
end

Base.show(io::IO, transfer::PeriodicTransferMatrix) = summary(io, transfer)

function Base.show(io::IO, ::MIME"text/plain", transfer::PeriodicTransferMatrix)
    summary(io, transfer)
    get(io, :compact, false)::Bool && return nothing
    count = length(transfer)
    limit = get(io, :limit, true)::Bool
    visible_half = limit ? max(1, div(displaysize(io)[1] - 4, 4)) : typemax(Int)
    show_all = !limit || count <= 2 * visible_half
    spaces = [TensorKit.domain(transfer[count]);
        [TensorKit.codomain(transfer[i]) for i in 2:count]]
    labels = map(spaces) do space
        length(space) == 2 ?
            (sprint(show, space[1]; context=io), sprint(show, space[2]; context=io)) :
            (sprint(show, space; context=io), "")
    end
    width = maximum(textwidth(left) for (left, _) in labels)
    indent = repeat(" ", width + 1)
    gap = repeat(" ", ndigits(count) + 6)
    skeleton = "│ │" * gap * "│ │"
    function bond(site)
        left, right = labels[site]
        return repeat(" ", width - textwidth(left)) * left * " " * skeleton *
            (isempty(right) ? "" : " " * right)
    end
    println(io, ":")
    println(io, indent, "╭─╮", gap, "╭─╮")
    println(io, bond(1))
    for site in count:-1:1
        if show_all || site > count - visible_half || site <= visible_half
            println(io, indent, "│ ├──[", lpad(site, ndigits(count)), "]──┤ │")
            site == 1 || println(io, bond(site))
        elseif site == visible_half + 1
            println(io, indent, skeleton, " ⋮")
        end
    end
    print(io, indent, "╰─╯", gap, "╰─╯")
    return nothing
end
