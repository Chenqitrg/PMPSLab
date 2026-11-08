using PMPSLab: P_decomp
# Run from PMPSLab: julia --project=. examples/gauge/02_projector_decomposition.jl
# Construct the two bond matrices from a pair of gauges, without an MPS.
using PMPSLab: P_decomp
using MPSKit: TruncationStrategy
using TensorKit: TensorMap, ℂ, id
using Test: @test

V = ℂ^2
R = TensorMap([4.0 0.0; 0.0 1.0], V, V)
L = TensorMap([1.0 0.0; 0.0 2.0], V, V)

# P_decomp uses the SVD of L * R and returns (PR, PL), in that order.
# TruncationStrategy() requests no truncation.
PR, PL = P_decomp(R, L, TruncationStrategy())
@test PR * PL ≈ id(Float64, V)
display(PR)
display(PL)

# This identity relies on a well-conditioned full-rank gauge and no truncation.
# Rank-deficient or truncated cases instead project onto a retained subspace.
# PR and PL are generally neither equal nor adjoints of one another.
nothing
