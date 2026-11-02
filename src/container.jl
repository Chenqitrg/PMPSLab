# Periodic MPS container and its closed-bond invariant.

"""
    mul_left_virtual(gauge, tensor)

Multiply `gauge` into the left virtual leg of an MPS tensor while preserving
the tensor's physical-leg partition and ordering.
"""
function mul_left_virtual(
        gauge::AbstractTensorMap{E,S,1,1},
        tensor::AbstractTensorMap{E,S,N,1},
    ) where {E,S,N}
    tail = repartition(tensor, 1, N)
    return repartition(gauge * tail, N, 1)
end

"""
    mul_right_virtual(tensor, gauge)

Multiply `gauge` into the right virtual leg of an MPS tensor.
"""
function mul_right_virtual(
        tensor::AbstractTensorMap{E,S,N,1},
        gauge::AbstractTensorMap{E,S,1,1},
    ) where {E,S,N}
    return tensor * gauge
end

function _assert_closed_bonds(tensors::PeriodicVector{<:GenericMPSTensor})
    isempty(tensors) && throw(ArgumentError(
        "a periodic MPS must contain at least one tensor"
    ))
    for site in eachindex(tensors)
        next_site = site + 1
        right = right_virtualspace(tensors[site])
        left = left_virtualspace(tensors[next_site])
        right == left || throw(TensorKit.SpaceMismatch(
            "right virtual space at site $site ($right) does not match " *
            "left virtual space at site $next_site ($left)"
        ))
    end
    return nothing
end

function _insert_gauge!(
        tensors::PeriodicVector{<:GenericMPSTensor},
        site::Integer,
        right_matrix::AbstractTensorMap{E,S,1,1},
        left_matrix::AbstractTensorMap{E,S,1,1},
    ) where {E,S}
    left_site = mod1(site, length(tensors))
    right_site = left_site + 1
    tensors[left_site] =
        mul_right_virtual(tensors[left_site], right_matrix)
    tensors[right_site] =
        mul_left_virtual(left_matrix, tensors[right_site])
    return tensors
end

function _standardize(tensors::PeriodicVector{<:GenericMPSTensor})
    _assert_closed_bonds(tensors)
    standardized = copy(tensors)
    for site in eachindex(standardized)
        bond = right_virtualspace(standardized[site])
        isdual(bond) || continue
        gauge = unitary(
            scalartype(standardized[site]),
            bond,
            flip(bond),
        )
        _insert_gauge!(
            standardized, site, gauge, adjoint(gauge)
        )
    end
    return standardized
end

struct PeriodicMPS{A<:GenericMPSTensor} <: MPSKit.AbstractMPS
    tensors::PeriodicVector{A}
    function PeriodicMPS(
            tensors::PeriodicVector{A},
        ) where {A<:GenericMPSTensor}
        mps = new{A}(_standardize(tensors))
        assertconsistency(mps)
        return mps
    end
end

function PeriodicMPS(tensors::AbstractVector{<:GenericMPSTensor})
    return PeriodicMPS(PeriodicVector(tensors))
end

"""
    assertconsistency(mps::PeriodicMPS)

Check that the virtual spaces of adjacent site tensors are compatible around
the period. Throws `TensorKit.SpaceMismatch` on failure.
"""
function assertconsistency(mps::PeriodicMPS)
    _assert_closed_bonds(parent(mps))
    for i in eachindex(mps)
        bond = right_virtualspace(mps[i])
        isdual(bond) && throw(TensorKit.SpaceMismatch(
            "right virtual space at site $i ($bond) is not standardized"
        ))
    end
    return nothing
end

Base.size(mps::PeriodicMPS, args...) = size(mps.tensors, args...)
Base.length(mps::PeriodicMPS) = length(mps.tensors)
Base.firstindex(mps::PeriodicMPS) = firstindex(mps.tensors)
Base.lastindex(mps::PeriodicMPS) = lastindex(mps.tensors)
Base.eachindex(mps::PeriodicMPS) = eachindex(mps.tensors)
Base.eachindex(l::IndexStyle, mps::PeriodicMPS) = eachindex(l, mps.tensors)
eachsite(mps::PeriodicMPS) = PeriodicVector(eachindex(mps))
Base.checkbounds(::Type{Bool}, mps::PeriodicMPS, i::Integer) = true

Base.getindex(mps::PeriodicMPS, i::Int) = mps.tensors[i]

function Base.setindex!(
        mps::PeriodicMPS,
        tensor::GenericMPSTensor,
        i::Int,
    )
    site = mod1(i, length(mps))
    left = left_virtualspace(tensor)
    right = right_virtualspace(tensor)
    (!isdual(left) && !isdual(right)) || throw(TensorKit.SpaceMismatch(
        "PeriodicMPS site tensors require standardized nondual virtual " *
        "spaces, got $left and $right"
    ))

    if length(mps) == 1
        left == right || throw(TensorKit.SpaceMismatch(
            "one-site periodic MPS requires matching left and right virtual " *
            "spaces, got $left and $right"
        ))
    else
        expected_left = right_virtualspace(mps[site - 1])
        expected_right = left_virtualspace(mps[site + 1])
        left == expected_left || throw(TensorKit.SpaceMismatch(
            "left_virtualspace at site $site ($left) does not match the " *
            "previous right virtual space ($expected_left)"
        ))
        right == expected_right || throw(TensorKit.SpaceMismatch(
            "right_virtualspace at site $site ($right) does not match the " *
            "next left virtual space ($expected_right)"
        ))
    end

    return mps.tensors[site] = tensor
end

Base.iterate(mps::PeriodicMPS, args...) = iterate(mps.tensors, args...)

Base.eltype(::Type{<:PeriodicMPS{A}}) where {A} = A
Base.eltype(mps::PeriodicMPS) = eltype(typeof(mps))
Base.isfinite(::Type{<:PeriodicMPS}) = true

struct PeriodicChainStyle <: MPSKit.GeometryStyle end

MPSKit.GeometryStyle(::Type{<:PeriodicMPS}) = PeriodicChainStyle()
MPSKit.site_type(::Type{<:PeriodicMPS{A}}) where {A} = A
scalartype(mps::PeriodicMPS) = scalartype(first(mps))

Base.parent(mps::PeriodicMPS) = mps.tensors
Base.copy(mps::PeriodicMPS) = PeriodicMPS(copy(parent(mps)))

"""
    map_tensors(f, mps::PeriodicMPS) -> PeriodicMPS

Apply `f` independently to every site tensor and return a new periodic MPS.
The original MPS and its tensor container are not modified. The transformed
tensors must still form a consistent closed chain; the [`PeriodicMPS`](@ref)
constructor validates and standardizes the result.

As with [`map`](@ref), tensor aliasing is determined by `f`: for example,
`map_tensors(identity, mps)` has a fresh container but retains the original
tensor objects, while `map_tensors(copy, mps)` also copies every tensor.

Mapping the underlying `PeriodicVector` directly preserves its periodic
container type. Besides retaining periodic indexing, this keeps the transform
as a functional tensor operation without first materializing an ordinary
`Vector`.
"""
function map_tensors(f, mps::PeriodicMPS)
    return PeriodicMPS(map(f, parent(mps)))
end

"""
    standardize(mps::PeriodicMPS)

Return a standardized copy of `mps`. Construction already standardizes every
virtual bond, so this operation is idempotent.
"""
standardize(mps::PeriodicMPS) = copy(mps)

"""
    insert_gauge!(mps::PeriodicMPS, site::Integer, right_matrix, left_matrix)

Insert two matrices on the bond to the right of `site` (between sites `site`
and `site + 1`, with periodic wrapping). Obtain its space with
`right_virtualspace(mps, site)`. If the current bond has space `V` and the new
bond has space `W`, `right_matrix` must have the form `V ← W` and
`left_matrix` the form `W ← V`. They are absorbed into the right virtual leg
to the left of the bond and the left virtual leg to its right, respectively.

The matrices may be rectangular and need not be adjoints or inverses. Thus the
operation can represent either a gauge transformation or projector absorption.
"""
function insert_gauge!(
        mps::PeriodicMPS,
        site::Integer,
        right_matrix::AbstractTensorMap{E,S,1,1},
        left_matrix::AbstractTensorMap{E,S,1,1},
    ) where {E,S}
    old_bond = right_virtualspace(mps, site)
    new_bond = domain(right_matrix)[1]
    isdual(new_bond) && throw(TensorKit.SpaceMismatch(
        "PeriodicMPS bond matrices require a standardized nondual target " *
        "space, got $new_bond"
    ))
    codomain(right_matrix)[1] == old_bond ||
        throw(TensorKit.SpaceMismatch(
            "right matrix codomain $(codomain(right_matrix)[1]) does not " *
            "match bond $old_bond to the right of site $site"
        ))
    domain(left_matrix)[1] == old_bond ||
        throw(TensorKit.SpaceMismatch(
            "left matrix domain $(domain(left_matrix)[1]) does not match " *
            "bond $old_bond to the right of site $site"
        ))
    codomain(left_matrix)[1] == new_bond ||
        throw(TensorKit.SpaceMismatch(
            "left matrix codomain $(codomain(left_matrix)[1]) does not " *
            "match new bond $new_bond"
        ))

    _insert_gauge!(
        parent(mps), site, right_matrix, left_matrix
    )
    assertconsistency(mps)
    return mps
end

function Base.copyto!(dst::PeriodicMPS, src::PeriodicMPS)
    length(dst) == length(src) || throw(DimensionMismatch(
        "destination has $(length(dst)) sites but source has $(length(src))"
    ))

    # Convert and validate the complete replacement before mutating `dst`.
    # The final copy cannot fail from element conversion or inconsistent bonds,
    # so a rejected source never leaves a partially updated destination.
    tensors = Vector{eltype(dst)}(undef, length(dst))
    copyto!(tensors, src)
    replacement = PeriodicMPS(tensors)
    copyto!(parent(dst), parent(replacement))
    return dst
end

Base.repeat(mps::PeriodicMPS, i::Int) = PeriodicMPS(repeat(mps.tensors, i))
Base.circshift(mps::PeriodicMPS, n) = PeriodicMPS(circshift(mps.tensors, n))
Base.similar(mps::PeriodicMPS) =
    PeriodicMPS([similar(tensor) for tensor in mps])

left_virtualspace(mps::PeriodicMPS, site::Integer) =
    left_virtualspace(mps[Int(site)])
right_virtualspace(mps::PeriodicMPS, site::Integer) =
    right_virtualspace(mps[Int(site)])
physicalspace(mps::PeriodicMPS, site::Integer) =
    physicalspace(mps[Int(site)])

physicalspace(mps::PeriodicMPS{<:GenericMPSTensor}) =
    physicalspace(first(mps))
physicalspaces(mps::PeriodicMPS{<:GenericMPSTensor}) =
    PeriodicVector(map(physicalspace, mps))
