"""
    _coarse_grain(mps)

Contract the eight-site, one-physical-leg periodic MPS produced by the LoopTNR
split into the next checkerboard tensor cell. The site ordering is the U-shaped
ordering established by `PeriodicMPS(::AbstractTensorCell)` followed by the
two-physical-leg tensor split.
"""
function _coarse_grain(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
    ) where {E,S}
    length(mps) == 8 || throw(DimensionMismatch(
        "LoopTNR coarse graining requires eight split sites, got $(length(mps))"
    ))

    @plansor white[-1 -2; -3 -4] :=
        mps[1][S W; -1] *
        mps[8][-2 E; S] *
        mps[4][-3 W; N] *
        mps[5][N E; -4]
    @plansor black[-1 -2; -3 -4] :=
        mps[2][-4 N; E] *
        mps[3][E S; -2] *
        mps[6][-1 S; W] *
        mps[7][W N; -3]

    return CheckerboardGates(black, white)
end
