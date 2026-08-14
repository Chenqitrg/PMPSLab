# ------------------------------------------------------------------------------
# Sweep schemes — scanning orders as stateful iterators
#
# Each scheme is an iterator yielding `SweepState` values:
#   site — the position to update next
#   type — the scan direction after the update
#   n    — completed lap count (internal stop-check)
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
    ZigzagSweep(; N, maximal_sweep)

Yields sites `1, 2, …, N, N-1, …, 2` per round trip, repeating up to
`maximal_sweep` trips. At least two sites are required; a one-site system has
no zigzag direction and should use [`CircularSweep`](@ref).
"""
@kwdef struct ZigzagSweep <: AbstractSweep
    N::Int
    maximal_sweep::Int
    function ZigzagSweep(N::Int, maximal_sweep::Int)
        N >= 2 || throw(ArgumentError(
            "ZigzagSweep requires at least two sites, got $N; " *
            "use CircularSweep for a one-site sweep"
        ))
        maximal_sweep >= 0 || throw(ArgumentError(
            "maximal_sweep must be non-negative, got $maximal_sweep"
        ))
        return new(N, maximal_sweep)
    end
end

sweep_domain(z::ZigzagSweep) = Base.OneTo(z.N)

Base.IteratorSize(::Type{<:ZigzagSweep}) = Base.HasLength()
Base.IteratorEltype(::Type{<:ZigzagSweep}) = Base.HasEltype()
Base.eltype(::Type{<:ZigzagSweep}) = SweepState

Base.length(z::ZigzagSweep) = (2z.N - 2) * z.maximal_sweep

Base.iterate(z::ZigzagSweep) = iterate(z, 0)

function Base.iterate(z::ZigzagSweep, step::Int)
    step == length(z) && return nothing
    period = 2z.N - 2
    n, phase = divrem(step, period)
    state = phase < z.N - 1 ?
        SweepState{Forward}(phase + 1, n) :
        SweepState{Backward}(period - phase + 1, n)
    return state, step + 1
end

# ---------------------------------------------------------------------------
# CircularSweep — 循环扫描
# ---------------------------------------------------------------------------

"""
    CircularSweep(; N, maximal_sweep)

Yields sites `1, 2, …, N` repeatedly, `maximal_sweep` laps in total.
`dir` is always +1.
"""
@kwdef struct CircularSweep <: AbstractSweep
    N::Int
    maximal_sweep::Int
    function CircularSweep(N::Int, maximal_sweep::Int)
        N > 0 || throw(ArgumentError("sweep length must be positive, got $N"))
        maximal_sweep >= 0 || throw(ArgumentError(
            "maximal_sweep must be non-negative, got $maximal_sweep"
        ))
        return new(N, maximal_sweep)
    end
end

sweep_domain(cs::CircularSweep) = Base.OneTo(cs.N)

Base.IteratorSize(::Type{<:CircularSweep}) = Base.HasLength()
Base.IteratorEltype(::Type{<:CircularSweep}) = Base.HasEltype()
Base.eltype(::Type{<:CircularSweep}) = SweepState
Base.length(cs::CircularSweep) = cs.N * cs.maximal_sweep

Base.iterate(cs::CircularSweep) = iterate(cs, 0)

function Base.iterate(cs::CircularSweep, step::Int)
    step == length(cs) && return nothing
    n, phase = divrem(step, cs.N)
    return SweepState{Forward}(phase + 1, n), step + 1
end
