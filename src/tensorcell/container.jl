const GenericGateTensor = AbstractTensorMap{E,S,2,2} where {E,S}

"""
    AbstractTensorCell{G} <: AbstractMatrix{G}

Periodic matrix unit cell of square four-leg gate tensors. Integer row and
column indices wrap around the stored unit cell.
"""
abstract type AbstractTensorCell{G<:GenericGateTensor} <: AbstractMatrix{G} end

Base.IndexStyle(::Type{<:AbstractTensorCell}) = IndexCartesian()
Base.checkbounds(
    ::Type{Bool}, ::AbstractTensorCell{G}, ::Integer, ::Integer
) where {G} = true

"""
    UniformGates(gate)

One-gate periodic 1×1 unit cell. Every lattice coordinate returns `gate`.
"""
struct UniformGates{G<:GenericGateTensor} <: AbstractTensorCell{G}
    gate::G
    function UniformGates(gate::G) where {G<:GenericGateTensor}
        cell = new{G}(gate)
        assertconsistency(cell)
        return cell
    end
end

Base.size(::UniformGates) = (1, 1)
Base.getindex(cell::UniformGates, ::Integer, ::Integer) = cell.gate

"""
    CheckerboardGates(black, white)
    CheckerboardGates(cell::UniformGates; copy=true)

Two-gate checkerboard unit cell. The stored row is `[black white]`; neighboring
rows are staggered as `[white black]`.
"""
struct CheckerboardGates{G<:GenericGateTensor} <: AbstractTensorCell{G}
    black::G
    white::G
    function CheckerboardGates(black::G, white::G) where {G<:GenericGateTensor}
        cell = new{G}(black, white)
        assertconsistency(cell)
        return cell
    end
end

Base.size(::CheckerboardGates) = (1, 2)
function Base.getindex(cell::CheckerboardGates, row::Integer, column::Integer)
    return iseven(row + column) ? cell.black : cell.white
end

_copy_gate(gate, copy::Bool) = copy ? Base.copy(gate) : gate

CheckerboardGates(cell::UniformGates; copy::Bool = true) = CheckerboardGates(
    _copy_gate(cell.gate, copy), _copy_gate(cell.gate, copy)
)

"""
    PlaquetteGates(gates::AbstractMatrix)
    PlaquetteGates(northwest, northeast, southwest, southeast)
    PlaquetteGates(cell::UniformGates; copy=true)
    PlaquetteGates(cell::CheckerboardGates; copy=true)

Arbitrary periodic 2×2 gate plaquette. Matrix coordinates follow Julia's
`[row, column]` convention, matching the 2×2 unit-cell input used by LoopTNR.
"""
struct PlaquetteGates{G<:GenericGateTensor} <: AbstractTensorCell{G}
    northwest::G
    northeast::G
    southwest::G
    southeast::G
    function PlaquetteGates(
            northwest::G,
            northeast::G,
            southwest::G,
            southeast::G,
        ) where {G<:GenericGateTensor}
        cell = new{G}(northwest, northeast, southwest, southeast)
        assertconsistency(cell)
        return cell
    end
end

function PlaquetteGates(gates::AbstractMatrix{G}) where {G<:GenericGateTensor}
    size(gates) == (2, 2) || throw(DimensionMismatch(
        "a gate plaquette must have size (2, 2), got $(size(gates))"
    ))
    return PlaquetteGates(
        gates[1, 1], gates[1, 2], gates[2, 1], gates[2, 2]
    )
end

PlaquetteGates(cell::UniformGates; copy::Bool = true) = PlaquetteGates(
    _copy_gate(cell.gate, copy),
    _copy_gate(cell.gate, copy),
    _copy_gate(cell.gate, copy),
    _copy_gate(cell.gate, copy),
)

PlaquetteGates(cell::CheckerboardGates; copy::Bool = true) = PlaquetteGates(
    _copy_gate(cell.black, copy),
    _copy_gate(cell.white, copy),
    _copy_gate(cell.white, copy),
    _copy_gate(cell.black, copy),
)

Base.size(::PlaquetteGates) = (2, 2)
function Base.getindex(cell::PlaquetteGates, row::Integer, column::Integer)
    wrapped_row = mod1(row, 2)
    wrapped_column = mod1(column, 2)
    if wrapped_row == 1
        return wrapped_column == 1 ? cell.northwest : cell.northeast
    end
    return wrapped_column == 1 ? cell.southwest : cell.southeast
end

"""
    assertconsistency(cell::AbstractTensorCell)

Check periodic nearest-neighbor spaces using the square-lattice leg convention
`(West, South; North, East)`. Uniform gates require `reverse(domain(gate)) ==
codomain(gate)`, while checkerboard gates require this relation across opposite
colors. Throws `TensorKit.SpaceMismatch` when a periodic bond does not close.
"""
function _assert_reverse_match(source, target, source_name, target_name)
    source_space = reverse(domain(source))
    target_space = codomain(target)
    source_space == target_space || throw(TensorKit.SpaceMismatch(
        "reversed domain of $source_name gate ($source_space) does not match " *
        "codomain of $target_name gate ($target_space)"
    ))
    return nothing
end

function assertconsistency(cell::UniformGates)
    return _assert_reverse_match(
        cell.gate, cell.gate, "uniform", "uniform"
    )
end

function assertconsistency(cell::CheckerboardGates)
    _assert_reverse_match(cell.black, cell.white, "black", "white")
    _assert_reverse_match(cell.white, cell.black, "white", "black")
    return nothing
end

function assertconsistency(cell::AbstractTensorCell)
    for column in axes(cell, 2), row in axes(cell, 1)
        gate = cell[row, column]
        east = domain(gate)[2]
        west = codomain(cell[row, column + 1])[1]
        east == west || throw(TensorKit.SpaceMismatch(
            "east space at ($row, $column) ($east) does not match west " *
            "space at ($row, $(mod1(column + 1, size(cell, 2)))) ($west)"
        ))

        south = codomain(gate)[2]
        north = domain(cell[row + 1, column])[1]
        south == north || throw(TensorKit.SpaceMismatch(
            "south space at ($row, $column) ($south) does not match north " *
            "space at ($(mod1(row + 1, size(cell, 1))), $column) ($north)"
        ))
    end
    return nothing
end
