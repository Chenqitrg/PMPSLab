# ------------------------------------------------------------------------------
# 2-site → 1-site splitting — inverse of pair_contraction
# ------------------------------------------------------------------------------

abstract type AbstractLoopSplit <: Algorithm end

"""
    split(mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, alg::AbstractLoopSplit)

Given a periodic MPS with 2 physical legs per site (3 codim legs), produce a
periodic MPS with 1 physical leg per site (2 codim legs) that best approximates
the loop state. Returns `(mps_1site, log)`.
"""
function split(mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
               alg::AbstractLoopSplit) where {E,S}
    return _split(mps, alg)
end
