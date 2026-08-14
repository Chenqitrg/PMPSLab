const GenericGateTensor = AbstractTensorMap{E,S,2,2} where {E,S}

"""
    AbstractTensorCell{G} <: AbstractMatrix{G}

Periodic matrix unit cell of square four-leg gate tensors. Integer row and
column indices wrap around the stored unit cell. Construction standardizes
every periodic bond to a nondual space through paired gauge transformations.
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
        raw = new{G}(gate)
        _assert_closed_cell_bonds(raw)
        gates = _standardize_cell_gates(raw)
        cell = new{G}(gates[1, 1])
        assertconsistency(cell)
        return cell
    end
end

Base.size(::UniformGates) = (1, 1)
Base.getindex(cell::UniformGates, ::Integer, ::Integer) = cell.gate
Base.copy(cell::UniformGates) = UniformGates(copy(cell.gate))

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
        raw = new{G}(black, white)
        _assert_closed_cell_bonds(raw)
        gates = _standardize_cell_gates(raw)
        cell = new{G}(gates[1, 1], gates[1, 2])
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

Base.copy(cell::CheckerboardGates) = CheckerboardGates(
    copy(cell.black), copy(cell.white)
)

"""
    PlaquetteGates(gates::AbstractMatrix; copy=true)
    PlaquetteGates(northwest, northeast, southwest, southeast)
    PlaquetteGates(cell::UniformGates; copy=true)
    PlaquetteGates(cell::CheckerboardGates; copy=true)
    PlaquetteGates(cell::PlaquetteGates; copy=true)

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
        raw = new{G}(northwest, northeast, southwest, southeast)
        _assert_closed_cell_bonds(raw)
        gates = _standardize_cell_gates(raw)
        cell = new{G}(gates[1, 1], gates[1, 2], gates[2, 1], gates[2, 2])
        assertconsistency(cell)
        return cell
    end
end

function PlaquetteGates(
        gates::AbstractMatrix{G}; copy::Bool = true
    ) where {G<:GenericGateTensor}
    size(gates) == (2, 2) || throw(DimensionMismatch(
        "a gate plaquette must have size (2, 2), got $(size(gates))"
    ))
    return PlaquetteGates(
        _copy_gate(gates[1, 1], copy),
        _copy_gate(gates[1, 2], copy),
        _copy_gate(gates[2, 1], copy),
        _copy_gate(gates[2, 2], copy),
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

PlaquetteGates(cell::PlaquetteGates; copy::Bool = true) =
    copy ? Base.copy(cell) : cell

Base.copy(cell::PlaquetteGates) = PlaquetteGates(
    copy(cell.northwest),
    copy(cell.northeast),
    copy(cell.southwest),
    copy(cell.southeast),
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

_stored_index(::UniformGates, ::Integer, ::Integer) = CartesianIndex(1, 1)
function _stored_index(::CheckerboardGates, row::Integer, column::Integer)
    return CartesianIndex(1, iseven(row + column) ? 1 : 2)
end
function _stored_index(::PlaquetteGates, row::Integer, column::Integer)
    return CartesianIndex(mod1(row, 2), mod1(column, 2))
end

"""
    mul_west(matrix, gate)

Absorb `matrix` into the west (first codomain) leg of `gate`. If the old west
space is `V` and the new space is `W`, `matrix` must have the form `W ← V`.
"""
function mul_west(
        matrix::AbstractTensorMap{E,S,1,1}, gate::GenericGateTensor
    ) where {E,S}
    @plansor result[-1 -2; -3 -4] :=
        matrix[-1; 1] * gate[1 -2; -3 -4]
    return result
end

"""
    mul_south(matrix, gate)

Absorb `matrix` into the south (second codomain) leg of `gate`. If the old
south space is `V` and the new space is `W`, `matrix` must have the form
`W ← V`.
"""
function mul_south(
        matrix::AbstractTensorMap{E,S,1,1}, gate::GenericGateTensor
    ) where {E,S}
    @plansor result[-1 -2; -3 -4] :=
        gate[-1 1; -3 -4] * matrix[-2; 1]
    return result
end

"""
    mul_north(gate, matrix)

Absorb `matrix` into the north (first domain) leg of `gate`. If the old north
space is `V` and the new space is `W`, `matrix` must have the form `V ← W`.
"""
function mul_north(
        gate::GenericGateTensor, matrix::AbstractTensorMap{E,S,1,1}
    ) where {E,S}
    @plansor result[-1 -2; -3 -4] :=
        gate[-1 -2; 1 -4] * matrix[1; -3]
    return result
end

"""
    mul_east(gate, matrix)

Absorb `matrix` into the east (second domain) leg of `gate`. If the old east
space is `V` and the new space is `W`, `matrix` must have the form `V ← W`.
"""
function mul_east(
        gate::GenericGateTensor, matrix::AbstractTensorMap{E,S,1,1}
    ) where {E,S}
    @plansor result[-1 -2; -3 -4] :=
        gate[-1 -2; -3 1] * matrix[1; -4]
    return result
end

function _standardize_horizontal_bond!(
        gates::AbstractMatrix{<:GenericGateTensor},
        source::CartesianIndex{2},
        target::CartesianIndex{2},
    )
    bond = domain(gates[source])[2]
    isdual(bond) || return gates
    gauge = unitary(scalartype(gates[source]), bond, flip(bond))
    gates[source] = mul_east(gates[source], gauge)
    gates[target] = mul_west(adjoint(gauge), gates[target])
    return gates
end

function _standardize_vertical_bond!(
        gates::AbstractMatrix{<:GenericGateTensor},
        source::CartesianIndex{2},
        target::CartesianIndex{2},
    )
    bond = codomain(gates[source])[2]
    isdual(bond) || return gates
    gauge = unitary(scalartype(gates[source]), bond, flip(bond))
    gates[source] = mul_south(adjoint(gauge), gates[source])
    gates[target] = mul_north(gates[target], gauge)
    return gates
end

function _standardize_cell_gates(cell::AbstractTensorCell)
    gates = collect(cell)
    for column in axes(cell, 2), row in axes(cell, 1)
        source = _stored_index(cell, row, column)
        east = _stored_index(cell, row, column + 1)
        south = _stored_index(cell, row + 1, column)
        _standardize_horizontal_bond!(gates, source, east)
        _standardize_vertical_bond!(gates, source, south)
    end
    return gates
end

"""
    assertconsistency(cell::AbstractTensorCell)

Check standardized periodic nearest-neighbor spaces using the square-lattice
leg convention `(West, South; North, East)`. Uniform gates require
`reverse(domain(gate)) == codomain(gate)`, while checkerboard gates require
this relation across opposite colors. Throws `TensorKit.SpaceMismatch` when a
periodic bond does not close or a leg retains a dual space.
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

function _assert_closed_cell_bonds(cell::UniformGates)
    return _assert_reverse_match(
        cell.gate, cell.gate, "uniform", "uniform"
    )
end

function _assert_closed_cell_bonds(cell::CheckerboardGates)
    _assert_reverse_match(cell.black, cell.white, "black", "white")
    _assert_reverse_match(cell.white, cell.black, "white", "black")
    return nothing
end

function _assert_closed_cell_bonds(cell::AbstractTensorCell)
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

function assertconsistency(cell::AbstractTensorCell)
    _assert_closed_cell_bonds(cell)
    leg_names = ("west", "south", "north", "east")
    for column in axes(cell, 2), row in axes(cell, 1)
        gate = cell[row, column]
        spaces = (codomain(gate)..., domain(gate)...)
        for (name, bond) in zip(leg_names, spaces)
            isdual(bond) || continue
            throw(TensorKit.SpaceMismatch(
                "$name space at ($row, $column) ($bond) is not standardized"
            ))
        end
    end
    return nothing
end

"""
    standardize(cell::AbstractTensorCell)

Return a standardized copy of `cell`. Construction already standardizes every
horizontal and vertical periodic bond, so this operation is idempotent.
"""
standardize(cell::AbstractTensorCell) = copy(cell)
