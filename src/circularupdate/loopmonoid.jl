# ------------------------------------------------------------------------------
# LoopMonoid — abstract interface for periodic loop products
# ------------------------------------------------------------------------------

abstract type LoopMonoid end

Base.length(v::LoopMonoid) = error("length not implemented for $(typeof(v))")
Base.getindex(v::LoopMonoid, i::Int) = error("getindex not implemented for $(typeof(v))")

range_prod(v::LoopMonoid, r::AbstractUnitRange) = reduce(*, (v[i] for i in r))

# ------------------------------------------------------------------------------
# SweepState — one step's position in a sweep
#
#   site — the position to update
#   dir  — cache slide direction after update (+1 forward, -1 backward)
#   n    — completed lap count (for stop-check against maximal_sweep)
# ------------------------------------------------------------------------------

struct SweepState
    site::Int
    dir::Int
    n::Int
end
