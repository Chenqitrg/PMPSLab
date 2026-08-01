# ------------------------------------------------------------------------------
# 2-site → 1-site splitting — inverse of pair_contraction
# ------------------------------------------------------------------------------

abstract type AbstractSolve <: Algorithm end   # 每 site 更新规则
abstract type AbstractSweep <: Algorithm end   # 扫描方案：Zigzag, Circular
abstract type AbstractLoopSplit <: Algorithm end  # Split = sweep + solve 的组合

struct LoopSplit{S<:AbstractSweep, V<:AbstractSolve} <: AbstractLoopSplit
    sweep::S
    solve::V
end

# split(mps, alg) defined in context.jl — needs SplitContext