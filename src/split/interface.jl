# ------------------------------------------------------------------------------
# 2-site → 1-site splitting — inverse of pair_contraction
# ------------------------------------------------------------------------------

abstract type AbstractAnsatz <: Algorithm end   # initial ansatz for the 2-site → 1-site splitting
abstract type AbstractSolve <: Algorithm end   # 每 site 更新规则
abstract type AbstractSweep <: Algorithm end   # 扫描方案：Zigzag, Circular
abstract type AbstractPeriodicSplit <: Algorithm end  # Split = sweep + solve 的组合

struct PeriodicSplit{A<:AbstractAnsatz, S<:AbstractSweep, V<:AbstractSolve} <: AbstractPeriodicSplit
    ansatz::A
    sweep::S
    solve::V
end

# split(mps, alg) defined in context.jl — needs SplitContext
