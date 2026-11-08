# Run from PMPSLab: julia --project=. examples/environments/05_cache_update.jl
# After a tensor change, notify the cache and repair it using the current MPS.
using PMPSLab: PeriodicMPS
using TensorKit: TensorMap
using PMPSLab: transfer_at, build_transfer_cache, rebuild_transfer_cache!
using PeriodicProductsLab: invalidate!, environment, StaleCacheError
using Test: @test, @test_throws

mps = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
cache = build_transfer_cache(mps, mps)
# Cache entries may be updated in place; copy the old environment for comparison.
before = copy(environment(cache, 1))

# Tensor spaces stay the same; both bra and ket of the self-transfer change.
mps[2] = 2 * mps[2]
# The cache does not monitor the MPS; explicitly report the changed transfer site.
invalidate!(cache, 2)
@test_throws StaleCacheError environment(cache, 1)

# Generate needed transfers from the current MPS, reusing valid prefixes/suffixes.
rebuild_transfer_cache!(cache, mps, mps)
E = [transfer_at(mps, site) for site in 1:4]
@test environment(cache, 1) ≈ E[2] * E[3] * E[4]
# Both bra and ket scale by 2, so this environment scales by 4.
@test TensorMap(environment(cache, 1)) ≈ 4 * TensorMap(before)
println("Updated tensor → invalidate! → rebuild_transfer_cache!: checks passed")
# Directional updates use PeriodicProductsLab.advance!; this example uses rebuild_transfer_cache!.
nothing
