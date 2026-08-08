# ------------------------------------------------------------------------------
# AbstractPeriodicProduct — ordered products with periodic indexing
# ------------------------------------------------------------------------------

abstract type AbstractPeriodicProduct end

range_product(v::AbstractPeriodicProduct, r::AbstractUnitRange) =
    mapfoldl(i -> v[i], *, r)

# ------------------------------------------------------------------------------
# SweepState — one step's position in a sweep
#
#   site — the position to update
#   dir  — the direction (+1/-1) in which the cache advances after the update
#   n    — completed period count (for stop-check against maximal_sweep)
# ------------------------------------------------------------------------------

struct SweepState
    site::Int
    dir::Int
    n::Int
end
