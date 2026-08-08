# ------------------------------------------------------------------------------
# Sweep schemes — scanning orders as stateful iterators
#
# Each scheme is an iterator yielding `SweepState` values:
#   site — the position to update next
#   dir  — the scan direction (+1/-1) after the update
#   n    — completed lap count (internal stop-check)
#
# Iteration state is an immutable `SweepState`, separate from the container,
# so the same scheme can be iterated repeatedly from the start.
# ------------------------------------------------------------------------------

# AbstractSweep defined in split/interface.jl

# ---------------------------------------------------------------------------
# ZigzagSweep — 往返扫描
# ---------------------------------------------------------------------------

"""
    ZigzagSweep(N; maximal_sweep)

Yields sites `1, 2, …, N, N-1, …, 2` per round trip, repeating up to
`maximal_sweep` trips. The turn at `N` flips `dir` to -1.
"""
struct ZigzagSweep <: AbstractSweep
    N::Int
    maximal_sweep::Int
    function ZigzagSweep(N::Int, maximal_sweep::Int)
        N > 0 || throw(ArgumentError("sweep length must be positive, got $N"))
        maximal_sweep >= 0 || throw(ArgumentError(
            "maximal_sweep must be non-negative, got $maximal_sweep"
        ))
        return new(N, maximal_sweep)
    end
end

ZigzagSweep(N::Int; maximal_sweep::Int) = ZigzagSweep(N, maximal_sweep)

Base.IteratorSize(::Type{<:ZigzagSweep}) = Base.HasLength()
Base.IteratorEltype(::Type{<:ZigzagSweep}) = Base.HasEltype()
Base.eltype(::Type{<:ZigzagSweep}) = SweepState

function Base.length(z::ZigzagSweep)
    z.N == 1 && return z.maximal_sweep
    return (2z.N - 2) * z.maximal_sweep
end

function Base.iterate(z::ZigzagSweep, st::SweepState = SweepState(1, 1, 0))
    st.n >= z.maximal_sweep && return nothing
    site, dir, n = st.site, st.dir, st.n
    if z.N == 1
        return (SweepState(1, 1, n), SweepState(1, 1, n + 1))
    end
    if site == z.N && dir == 1
        # 峰值：掉头，扫描方向翻为 -1
        return (SweepState(site, -1, n), SweepState(z.N - 1, -1, n))
    elseif site == 2 && dir == -1
        # 一趟结束（不重访 site 1）
        return (SweepState(site, dir, n), SweepState(1, 1, n + 1))
    elseif dir == 1
        return (SweepState(site, dir, n), SweepState(site + 1, 1, n))
    else
        return (SweepState(site, dir, n), SweepState(site - 1, -1, n))
    end
end

# ---------------------------------------------------------------------------
# CircularSweep — 循环扫描
# ---------------------------------------------------------------------------

"""
    CircularSweep(N; maximal_sweep)

Yields sites `1, 2, …, N` repeatedly, `maximal_sweep` laps in total.
`dir` is always +1.
"""
struct CircularSweep <: AbstractSweep
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

CircularSweep(N::Int; maximal_sweep::Int) = CircularSweep(N, maximal_sweep)

Base.IteratorSize(::Type{<:CircularSweep}) = Base.HasLength()
Base.IteratorEltype(::Type{<:CircularSweep}) = Base.HasEltype()
Base.eltype(::Type{<:CircularSweep}) = SweepState
Base.length(cs::CircularSweep) = cs.N * cs.maximal_sweep

function Base.iterate(cs::CircularSweep, st::SweepState = SweepState(1, 1, 0))
    st.n >= cs.maximal_sweep && return nothing
    next_site = mod1(st.site + 1, cs.N)
    next_n = next_site == 1 ? st.n + 1 : st.n
    return (st, SweepState(next_site, 1, next_n))
end
