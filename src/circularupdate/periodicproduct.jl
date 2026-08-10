# ------------------------------------------------------------------------------
# AbstractPeriodicProduct — ordered products with periodic indexing
# ------------------------------------------------------------------------------

abstract type AbstractPeriodicProduct end

range_product(v::AbstractPeriodicProduct, r::AbstractUnitRange) =
    mapfoldl(i -> v[i], *, r)

# ------------------------------------------------------------------------------
# SweepState — one typed step's position in a sweep
#
#   site — the position to update
#   D    — the scan direction, encoded in the state type
#   n    — completed period count (for stop-check against maximal_sweep)
# ------------------------------------------------------------------------------

abstract type SweepDirection end
struct Forward <: SweepDirection end
struct Backward <: SweepDirection end

struct SweepState{D<:SweepDirection}
    site::Int
    n::Int
end
