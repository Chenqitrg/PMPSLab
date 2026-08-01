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

"""
    split(mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, alg::AbstractLoopSplit)

Given a periodic MPS with 2 physical legs per site (3 codim legs), produce a
periodic MPS with 1 physical leg per site (2 codim legs) that best approximates
the loop state. Returns `(mps_1site, log)`.
"""
function split(mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
               alg::LoopSplit) where {E,S}
    return _split(mps, alg)
end
