# Display is local to the value type; no periodic-cache rendering dependencies.
function Base.summary(io::IO, transfer::DenseTransferMatrix)
    data = TensorKit.TensorMap(transfer)
    print(io, TensorKit.dim(TensorKit.codomain(data)), "×",
        TensorKit.dim(TensorKit.domain(data)), " DenseTransferMatrix(",
        string(TensorKit.scalartype(data)), ", ",
        TensorKit.type_repr(TensorKit.spacetype(data)), ")")
    return nothing
end

Base.show(io::IO, transfer::DenseTransferMatrix) = summary(io, transfer)

function Base.show(io::IO, ::MIME"text/plain", transfer::DenseTransferMatrix)
    summary(io, transfer)
    get(io, :compact, false)::Bool && return nothing
    data = TensorKit.TensorMap(transfer)
    TensorKit.numout(data) == 2 && TensorKit.numin(data) == 2 || return nothing
    top = TensorKit.domain(data)
    bottom = TensorKit.codomain(data)
    labels = [(sprint(show, s[1]; context=io), sprint(show, s[2]; context=io))
        for s in (top, bottom)]
    width = maximum(textwidth(left) for (left, _) in labels)
    bond((left, right)) = repeat(" ", width - textwidth(left)) *
        left * " │       │ " * right
    println(io, ":")
    println(io, bond(labels[1]))
    println(io, repeat(" ", width + 1), "├───────┤")
    print(io, bond(labels[2]))
    return nothing
end
