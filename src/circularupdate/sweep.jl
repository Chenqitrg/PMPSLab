# ------------------------------------------------------------------------------
# Sweep schemes — scanning orders as stateful iterators
#
# Each scheme is a mutable iterator yielding `(site, dir)` pairs:
#   site — the position to update next
#   dir  — the direction (+1/-1) in which the cache cut slides after the update
# Both stop on their own: Zip after one round trip, Circular after
# `maximal_sweep` full laps (n_sweep counts completed laps).
# ------------------------------------------------------------------------------

abstract type AbstractSweep <: AbstractLoopApproximation end

"""
    ZipSweep(N) — 往返（拉链式）扫描

Yields sites `1, 2, …, N, N-1, …, 2` once, then ends. The turn at `N` flips
`dir` to -1.
"""
@kwdef mutable struct ZipSweep <: AbstractSweep
    N::Int
    site::Int = 1
    dir::Int = 1
    turned::Bool = false
end

ZipSweep(N::Int) = ZipSweep(N, 1, 1, false)

Base.IteratorSize(::Type{<:ZipSweep}) = Base.HasLength()
Base.IteratorEltype(::Type{<:ZipSweep}) = Base.HasEltype()
Base.eltype(::Type{<:ZipSweep}) = Tuple{Int, Int}
Base.length(z::ZipSweep) = z.N == 1 ? 1 : 2z.N - 2

function Base.iterate(z::ZipSweep, state = z)
    z.turned && return nothing
    site, dir = z.site, z.dir
    next = site + dir
    if next > z.N
        dir = -1
        next = z.N - 1
        next < 1 && (z.turned = true)   # N = 1：只访问一次
    elseif next < 1
        z.turned = true                 # 回到起点，结束（不重复访问 1）
        next = 1
    end
    z.site, z.dir = next, dir
    return ((site, dir), state)
end

"""
    CircularSweep(N; maximal_sweep = 40) — 循环扫描

Yields sites `1, 2, …, N` repeatedly, `maximal_sweep` laps in total; `n_sweep`
counts completed laps. `dir` is always +1.
"""
@kwdef mutable struct CircularSweep <: AbstractSweep
    N::Int
    maximal_sweep::Int = 40
    n_sweep::Int = 0
    site::Int = 1
end

CircularSweep(N::Int; maximal_sweep::Int = 40) =
    CircularSweep(N, maximal_sweep, 0, 1)

Base.IteratorSize(::Type{<:CircularSweep}) = Base.HasLength()
Base.IteratorEltype(::Type{<:CircularSweep}) = Base.HasEltype()
Base.eltype(::Type{<:CircularSweep}) = Tuple{Int, Int}
Base.length(cs::CircularSweep) = cs.N * cs.maximal_sweep

function Base.iterate(cs::CircularSweep, state = cs)
    cs.n_sweep >= cs.maximal_sweep && return nothing
    site = cs.site
    cs.site = mod1(cs.site + 1, cs.N)
    cs.site == 1 && (cs.n_sweep += 1)
    return ((site, 1), state)
end
