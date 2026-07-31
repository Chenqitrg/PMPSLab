# ------------------------------------------------------------------------------
# LoopMonoid — abstract interface for periodic loop products
# ------------------------------------------------------------------------------

abstract type LoopMonoid end

Base.length(v::LoopMonoid) = error("length not implemented for $(typeof(v))")
Base.getindex(v::LoopMonoid, i::Int) = error("getindex not implemented for $(typeof(v))")

range_prod(v::LoopMonoid, r::AbstractUnitRange) = reduce(*, (v[i] for i in r))
