# Periodic MPS container and its closed-bond invariant.

struct PeriodicMPS{A<:GenericMPSTensor} <: MPSKit.AbstractMPS
    tensors::PeriodicVector{A}
    function PeriodicMPS(
            tensors::PeriodicVector{A},
        ) where {A<:GenericMPSTensor}
        isempty(tensors) && throw(ArgumentError(
            "a periodic MPS must contain at least one tensor"
        ))
        mps = new{A}(tensors)
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
    for i in eachindex(mps)
        L = left_virtualspace(mps[i])
        R = right_virtualspace(mps[i - 1])
        L == R || throw(TensorKit.SpaceMismatch(
            "left_virtualspace at site $i ($L) ≠ right_virtualspace at site $(i-1) ($R)"
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

Base.getindex(mps::PeriodicMPS, pos::HalfInt) =
    right_virtualspace(mps[pos.n])

virtualspace(mps::PeriodicMPS{<:GenericMPSTensor}) =
    left_virtualspace(first(mps))
virtualspaces(mps::PeriodicMPS{<:GenericMPSTensor}) =
    HalfPeriodicVector(map(right_virtualspace, mps))

physicalspace(mps::PeriodicMPS{<:GenericMPSTensor}) =
    physicalspace(first(mps))
physicalspaces(mps::PeriodicMPS{<:GenericMPSTensor}) =
    PeriodicVector(map(physicalspace, mps))
