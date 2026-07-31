# ------------------------------------------------------------------------------
# 2-to-1 site approximation — convert TwoSite MPS → OneSite MPS
# ------------------------------------------------------------------------------

"""
    approximate(mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, alg::AbstractLoopApproximation)

Given a periodic MPS with 2 physical legs per site (3 codim legs), produce a
periodic MPS with 1 physical leg per site (2 codim legs) that best approximates
the loop state. Returns `(mps_1site, log)`.
"""
function approximate(mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
                     alg::AbstractLoopApproximation) where {E,S}
    return _approximate(mps, alg)
end
