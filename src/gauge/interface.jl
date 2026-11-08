# ------------------------------------------------------------------------------
# Canonicalization interface
# ------------------------------------------------------------------------------

# Absorb the left and right bond projectors without assuming a fixed number of
# physical legs: T_i <- PL[i-1] * T_i * PR[i].
# Both projector vectors use the bond to the right of site i.
function _absorb_projectors!(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}},
        PR_vec::PeriodicVector{<:AbstractTensorMap{E,S,1,1}},
        PL_vec::PeriodicVector{<:AbstractTensorMap{E,S,1,1}}
    ) where {E,S,N}
    for site in eachindex(mps)
        insert_gauge!(mps, site, PR_vec[site], PL_vec[site])
    end
    return mps
end

"""
    gaugefix!(mps::PeriodicMPS, algorithm::ProjectorGaugeFix)

Use MPSKit's periodic left/right canonicalization to obtain normalized bond
gauges, construct the bond projectors, and absorb them into `mps`. Modifies and
returns `mps`. Before entering MPSKit's fixed-workspace canonicalization,
structurally incompatible virtual spaces are reduced exactly with local QR/LQ
transformations. The internal MPSKit solvers run without native log output.
"""
function gaugefix!(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}},
        algorithm::ProjectorGaugeFix,
    ) where {E,S,N}
    MPSKit.makefullrank!(
        mps.tensors;
        alg_orth=algorithm.orthogonalization_algorithm,
    )
    assertconsistency(mps)
    PR_vec, PL_vec = find_projectors(mps, algorithm)
    return _absorb_projectors!(mps, PR_vec, PL_vec)
end
