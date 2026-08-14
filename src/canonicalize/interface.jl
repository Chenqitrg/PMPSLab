# ------------------------------------------------------------------------------
# Canonicalization interface
# ------------------------------------------------------------------------------

# Absorb the left and right bond projectors without assuming a fixed number of
# physical legs: T_i <- PL_i * T_i * PR_i.
function _absorb_projectors!(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}},
        PR_vec::HalfPeriodicVector{<:AbstractTensorMap{E,S,1,1}},
        PL_vec::HalfPeriodicVector{<:AbstractTensorMap{E,S,1,1}}
    ) where {E,S,N}
    for site in eachindex(mps)
        bond = half(2site + 1)
        insert_gauge!(mps, bond, PR_vec[bond], PL_vec[bond])
    end
    return mps
end

"""
    gaugefix!(mps::PeriodicMPS, alg::ProjectorCanonical)

Use MPSKit's periodic left/right canonicalization to obtain normalized bond
gauges, construct the bond projectors, and absorb them into `mps`. Modifies and
returns `mps`. Before entering MPSKit's fixed-workspace canonicalization,
structurally incompatible virtual spaces are reduced exactly with local QR/LQ
transformations. The internal MPSKit solvers run without native log output.
"""
function gaugefix!(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}}, alg::ProjectorCanonical
    ) where {E,S,N}
    MPSKit.makefullrank!(mps.tensors; alg_orth = alg.alg_orth)
    assertconsistency(mps)
    PR_vec, PL_vec = find_projectors(mps, alg)
    return _absorb_projectors!(mps, PR_vec, PL_vec)
end
