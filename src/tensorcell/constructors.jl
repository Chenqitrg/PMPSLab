"""
    rand([rng], UniformGates, elt, space::TensorMapSpace)

Generate a uniform gate cell in `space`. The space must satisfy
`reverse(domain(space)) == codomain(space)`.
"""
function Random.rand(
        rng::Random.AbstractRNG,
        ::Type{UniformGates},
        elt::Type{<:Number},
        space::TensorMapSpace,
    )
    return UniformGates(Random.rand(rng, elt, space))
end

function Random.rand(
        ::Type{UniformGates},
        elt::Type{<:Number},
        space::TensorMapSpace,
    )
    return Random.rand(Random.default_rng(), UniformGates, elt, space)
end

"""
    rand([rng], CheckerboardGates, elt, blackspace::TensorMapSpace)

Generate the black gate in `blackspace` and the white gate in the uniquely
compatible checkerboard space.
"""
function Random.rand(
        rng::Random.AbstractRNG,
        ::Type{CheckerboardGates},
        elt::Type{<:Number},
        blackspace::TensorMapSpace,
    )
    whitespace = reverse(domain(blackspace)) ← reverse(codomain(blackspace))
    return CheckerboardGates(
        Random.rand(rng, elt, blackspace),
        Random.rand(rng, elt, whitespace),
    )
end

function Random.rand(
        ::Type{CheckerboardGates},
        elt::Type{<:Number},
        blackspace::TensorMapSpace,
    )
    return Random.rand(
        Random.default_rng(), CheckerboardGates, elt, blackspace
    )
end

"""
    rand([rng], PlaquetteGates, elt, spaces::AbstractMatrix{<:TensorMapSpace})

Generate a 2×2 plaquette from its gate spaces. The spaces must close on all
periodic nearest-neighbor bonds.
"""
function Random.rand(
        rng::Random.AbstractRNG,
        ::Type{PlaquetteGates},
        elt::Type{<:Number},
        spaces::AbstractMatrix{<:TensorMapSpace},
    )
    size(spaces) == (2, 2) || throw(DimensionMismatch(
        "a gate plaquette must have size (2, 2), got $(size(spaces))"
    ))
    gates = map(spaces) do space
        Random.rand(rng, elt, space)
    end
    return PlaquetteGates(gates; copy=false)
end

function Random.rand(
        ::Type{PlaquetteGates},
        elt::Type{<:Number},
        spaces::AbstractMatrix{<:TensorMapSpace},
    )
    return Random.rand(Random.default_rng(), PlaquetteGates, elt, spaces)
end
