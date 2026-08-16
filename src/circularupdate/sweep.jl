# ------------------------------------------------------------------------------
# Sweep schemes — scanning orders as stateful iterators
#
# Each scheme is an iterator yielding `SweepState` values:
#   site — the position to update next
#   type — the scan direction after the update
#   completed_sweeps — completed lap count (internal stop-check)
#
# Iteration state is a monotonic integer cursor. A yielded `SweepState` is
# derived from that cursor and never reused as mutable control state.
# ------------------------------------------------------------------------------

# AbstractSweep defined in split/interface.jl

"""
    sweep_domain(sweep::AbstractSweep)

Return the complete site domain covered by `sweep`. Custom sweep types must
define this method so algorithms can validate the sweep against their
numerical state before the first update.
"""
function sweep_domain(sweep::AbstractSweep)
    throw(ArgumentError(
        "$(typeof(sweep)) must define PMPSLab.sweep_domain"
    ))
end

# ---------------------------------------------------------------------------
# ZigzagSweep — 往返扫描
# ---------------------------------------------------------------------------

"""
    ZigzagSweep(; site_count, max_sweeps)

Yields sites `1, 2, …, site_count, site_count-1, …, 2` per round trip,
repeating up to `max_sweeps` trips. At least two sites are required; a
single-site system has no zigzag direction and should use
[`CircularSweep`](@ref).
"""
@kwdef struct ZigzagSweep <: AbstractSweep
    site_count::Int
    max_sweeps::Int
    function ZigzagSweep(site_count::Int, max_sweeps::Int)
        site_count >= 2 || throw(ArgumentError(
            "ZigzagSweep requires at least two sites, got $site_count; " *
            "use CircularSweep for a single-site sweep"
        ))
        max_sweeps >= 0 || throw(ArgumentError(
            "max_sweeps must be non-negative, got $max_sweeps"
        ))
        return new(site_count, max_sweeps)
    end
end

sweep_domain(z::ZigzagSweep) = Base.OneTo(z.site_count)

Base.IteratorSize(::Type{<:ZigzagSweep}) = Base.HasLength()
Base.IteratorEltype(::Type{<:ZigzagSweep}) = Base.HasEltype()
Base.eltype(::Type{<:ZigzagSweep}) = SweepState

Base.length(z::ZigzagSweep) = (2z.site_count - 2) * z.max_sweeps

Base.iterate(z::ZigzagSweep) = iterate(z, 0)

function Base.iterate(z::ZigzagSweep, step::Int)
    step == length(z) && return nothing
    period = 2z.site_count - 2
    completed_sweeps, phase = divrem(step, period)
    state = phase < z.site_count - 1 ?
        SweepState{Forward}(phase + 1, completed_sweeps) :
        SweepState{Backward}(period - phase + 1, completed_sweeps)
    return state, step + 1
end

# ---------------------------------------------------------------------------
# CircularSweep — 循环扫描
# ---------------------------------------------------------------------------

"""
    CircularSweep(; site_count, max_sweeps)

Yields sites `1, 2, …, site_count` repeatedly, `max_sweeps` laps in total. The
direction encoded in every yielded [`SweepState`](@ref) is [`Forward`](@ref).
"""
@kwdef struct CircularSweep <: AbstractSweep
    site_count::Int
    max_sweeps::Int
    function CircularSweep(site_count::Int, max_sweeps::Int)
        site_count > 0 || throw(ArgumentError(
            "sweep site_count must be positive, got $site_count"
        ))
        max_sweeps >= 0 || throw(ArgumentError(
            "max_sweeps must be non-negative, got $max_sweeps"
        ))
        return new(site_count, max_sweeps)
    end
end

sweep_domain(cs::CircularSweep) = Base.OneTo(cs.site_count)

Base.IteratorSize(::Type{<:CircularSweep}) = Base.HasLength()
Base.IteratorEltype(::Type{<:CircularSweep}) = Base.HasEltype()
Base.eltype(::Type{<:CircularSweep}) = SweepState
Base.length(cs::CircularSweep) = cs.site_count * cs.max_sweeps

Base.iterate(cs::CircularSweep) = iterate(cs, 0)

function Base.iterate(cs::CircularSweep, step::Int)
    step == length(cs) && return nothing
    completed_sweeps, phase = divrem(step, cs.site_count)
    return SweepState{Forward}(phase + 1, completed_sweeps), step + 1
end
