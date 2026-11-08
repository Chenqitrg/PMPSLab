# Run from PMPSLab: julia --project=. examples/environments/07_norm_and_normalization.jl
# Norms, distances, and normalization with or without modifying the input.
using PMPSLab: one_mps
using PMPSLab: squared_norm, squared_distance, normalize, normalize!
using LinearAlgebra: norm
using Test: @test

mps = one_mps(ComplexF64, 3, 2, 1)
@test squared_norm(mps) ≈ 8
@test norm(mps) ≈ sqrt(8)

# normalize returns a new MPS and distributes the scale evenly across all sites.
unit_mps = normalize(mps)
@test unit_mps !== mps
@test norm(unit_mps) ≈ 1
@test squared_norm(mps) ≈ 8

# squared_distance returns ||mps - unit_mps||², rather than the distance itself.
@test squared_distance(mps, unit_mps) ≈ (sqrt(8) - 1)^2
@show squared_distance(mps, unit_mps)

# The ! variant modifies the original MPS and returns that same object.
@test normalize!(mps) === mps
@test norm(mps) ≈ 1
@show norm(mps)
nothing
