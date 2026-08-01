# ------------------------------------------------------------------------------
# Sweep schemes — scanning orders as stateful iterators
#
# Each scheme is an iterator yielding `(site, dir)` pairs:
#   site — the position to update next
#   dir  — the direction (+1/-1) in which the cache cut slides after the update
#
# Iteration state is an immutable tuple, separate from the container, so the
# same scheme can be iterated repeatedly from the start.
# ------------------------------------------------------------------------------

abstract type AbstractSweep <: AbstractLoopSplit end

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
end

ZigzagSweep(N::Int; maximal_sweep::Int) = ZigzagSweep(N, maximal_sweep)

Base.IteratorSize(::Type{<:ZigzagSweep}) = Base.HasLength()
Base.IteratorEltype(::Type{<:ZigzagSweep}) = Base.HasEltype()
Base.eltype(::Type{<:ZigzagSweep}) = Tuple{Int, Int}

function Base.length(z::ZigzagSweep)
    z.N == 1 && return z.maximal_sweep
    return (2z.N - 2) * z.maximal_sweep
end

function Base.iterate(z::ZigzagSweep, (site, dir, n) = (1, 1, 0))
    n >= z.maximal_sweep && return nothing
    if z.N == 1
        return ((1, 1), (1, 1, n + 1))
    end
    if site == z.N && dir == 1
        # 峰值：掉头，slide 方向翻为 -1
        return ((site, -1), (z.N - 1, -1, n))
    elseif site == 2 && dir == -1
        # 一趟结束（不重访 site 1）
        return ((site, dir), (1, 1, n + 1))
    elseif dir == 1
        return ((site, dir), (site + 1, 1, n))
    else
        return ((site, dir), (site - 1, -1, n))
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
end

CircularSweep(N::Int; maximal_sweep::Int) = CircularSweep(N, maximal_sweep)

Base.IteratorSize(::Type{<:CircularSweep}) = Base.HasLength()
Base.IteratorEltype(::Type{<:CircularSweep}) = Base.HasEltype()
Base.eltype(::Type{<:CircularSweep}) = Tuple{Int, Int}
Base.length(cs::CircularSweep) = cs.N * cs.maximal_sweep

function Base.iterate(cs::CircularSweep, (site, n) = (1, 0))
    n >= cs.maximal_sweep && return nothing
    next_site = mod1(site + 1, cs.N)
    next_n = next_site == 1 ? n + 1 : n
    return ((site, 1), (next_site, next_n))
end
