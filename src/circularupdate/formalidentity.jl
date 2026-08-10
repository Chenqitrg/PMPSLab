"""
    FormalIdentity{T,S}

Formal identity for factor type `T` at an interface `space`. It represents an
empty ordered product without constructing an identity tensor. Composition
skips multiplication but still checks the domain/codomain compatibility that
multiplication would have enforced. `T` occupies no runtime storage.
"""
struct FormalIdentity{T,S}
    space::S
end

FormalIdentity{T}(space::S) where {T,S} = FormalIdentity{T,S}(space)

"""An element represented either by a concrete value or a formal identity."""
const UnitalElement{T,S} = Union{FormalIdentity{T,S},T}

"""
    edge_identity(v::AbstractPeriodicProduct{T})

Return the formal identity at the periodic edge without materializing a factor
or performing a factorization.
"""
edge_identity(v::AbstractPeriodicProduct{T}) where {T} =
    FormalIdentity{T}(edge_space(v))

"""
    compose_product(left, right)

Compose two unital elements without constructing or multiplying a dense
identity. Only composition of two concrete elements calls `*`.
"""
function compose_product(
        left::FormalIdentity{T},
        right::FormalIdentity{T},
    ) where {T}
    left.space == right.space || throw(SpaceMismatch(
        "formal-identity spaces do not match: $(left.space) ≠ $(right.space)"
    ))
    return left
end

function compose_product(left::FormalIdentity{T}, right::T) where {T}
    left.space == codomain(right) || throw(SpaceMismatch(
        "formal-identity space $(left.space) does not match right codomain $(codomain(right))"
    ))
    return right
end

function compose_product(left::T, right::FormalIdentity{T}) where {T}
    domain(left) == right.space || throw(SpaceMismatch(
        "left domain $(domain(left)) does not match formal-identity space $(right.space)"
    ))
    return left
end

compose_product(left::T, right::T) where {T} = left * right
