# Run from PMPSLab: julia --project=. examples/environments/04_cache.jl
# Cache transfer products, with explicit factors or the PMPS-specific adapter.
using PMPSLab: PeriodicMPS
using PMPSLab: transfer_at, PeriodicTransferMatrix, build_transfer_cache
using PeriodicProductsLab: build_cache, left_cache_before, right_cache_after, environment
using Test: @test

mps = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
transfers = PeriodicTransferMatrix(mps)
cache = build_cache(transfers)
# This named adapter generates transfers directly, without a separate collection.
direct = build_transfer_cache(mps, mps)

# E is only a reference for the checks below.
E = [transfer_at(mps, site) for site in 1:4]
@test left_cache_before(cache, 3) ≈ E[1] * E[2]
@test right_cache_after(cache, 2) ≈ E[3] * E[4]

# The environment of site 2 excludes E₂ and starts at the next site: E₃ E₄ E₁.
@test environment(cache, 2) ≈ E[3] * E[4] * E[1]
@test environment(direct, 2) ≈ environment(cache, 2)
display(environment(cache, 2))

# build_cache and environment belong to PeriodicProductsLab.
# build_transfer_cache belongs to PMPSLab and accepts PeriodicMPS.
nothing
