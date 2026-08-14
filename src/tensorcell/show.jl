function Base.summary(io::IO, cell::AbstractTensorCell)
    gate = cell[1, 1]
    print(
        io,
        size(cell, 1),
        "×",
        size(cell, 2),
        " ",
        nameof(typeof(cell)),
        "(",
        scalartype(gate),
        ", ",
        TensorKit.type_repr(spacetype(gate)),
        ")",
    )
    return nothing
end

Base.show(io::IO, cell::AbstractTensorCell) = summary(io, cell)

_display_rows(cell::AbstractTensorCell) = size(cell, 1)
_display_rows(::CheckerboardGates) = 2

function _vertical_cell_line(column_count::Int, marker::AbstractString)
    first_center = 5
    center_distance = 10
    return repeat(" ", first_center - 1) *
        join(fill(marker, column_count), repeat(" ", center_distance - 1))
end

function _tensor_cell_row(row::Int, column_count::Int)
    labels = ["[$row,$column]" for column in 1:column_count]
    return "…─" * join(labels, "─────") * "─…"
end

function _show_gate_spaces(io::IO, cell::AbstractTensorCell, row::Int, column::Int)
    gate = cell[row, column]
    print(
        io,
        "  [", row, ",", column, "] ",
        "W=", codomain(gate)[1],
        ", S=", codomain(gate)[2],
        ", N=", domain(gate)[1],
        ", E=", domain(gate)[2],
    )
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", cell::AbstractTensorCell)
    summary(io, cell)
    get(io, :compact, false)::Bool && return nothing

    println(io, ":")
    row_count = _display_rows(cell)
    column_count = size(cell, 2)
    continuation = _vertical_cell_line(column_count, "⋮")
    vertical_bonds = _vertical_cell_line(column_count, "│")

    println(io, continuation)
    println(io, vertical_bonds)
    for row in 1:row_count
        println(io, _tensor_cell_row(row, column_count))
        println(io, vertical_bonds)
    end
    println(io, continuation)

    for row in 1:row_count, column in 1:column_count
        _show_gate_spaces(io, cell, row, column)
        (row, column) == (row_count, column_count) || println(io)
    end
    return nothing
end
