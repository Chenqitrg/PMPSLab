# ------------------------------------------------------------------------------
# Canonicalization interface
# ------------------------------------------------------------------------------

# Absorb the left and right bond projectors without assuming a fixed number of
# physical legs: T_i <- PL_i * T_i * PR_i.
function _absorb_projectors!(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}},
        PL_vec::HalfPeriodicVector{<:AbstractTensorMap{E,S,1,1}},
        PR_vec::HalfPeriodicVector{<:AbstractTensorMap{E,S,1,1}}
    ) where {E,S,N}
    for i in 1:length(mps)
        left_bond = HalfInt(i - 1)
        right_bond = HalfInt(i)
        mps[i] = _projector_mul_front(PL_vec[left_bond], mps[i]) *
                 PR_vec[right_bond]
    end
    return mps
end

"""
    gaugefix!(mps::PeriodicMPS, alg::ProjectorCanonical)

Use MPSKit's periodic left/right canonicalization to obtain normalized bond
gauges, construct the bond projectors, and absorb them into `mps`. Modifies and
returns `mps`. The internal MPSKit solvers run without native log output.
"""
function gaugefix!(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}}, alg::ProjectorCanonical
    ) where {E,S,N}
    PR_vec, PL_vec = find_projectors(mps, alg)
    return _absorb_projectors!(mps, PL_vec, PR_vec)
end
