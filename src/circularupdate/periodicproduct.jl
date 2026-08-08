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
#   dir  — the scan direction (+1/-1), used to refresh the directional cache
#   n    — completed period count (for stop-check against maximal_sweep)
# ------------------------------------------------------------------------------

struct SweepState
    site::Int
    dir::Int
    n::Int
end
