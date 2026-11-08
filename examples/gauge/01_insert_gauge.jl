# Run from PMPSLab: julia --project=. examples/gauge/01_insert_gauge.jl
# PMPSLab owns the operation that absorbs two matrices into a virtual bond.
using PMPSLab: PeriodicMPS, dense, insert_gauge!, assertconsistency
using MPSKit: right_virtualspace
using TensorKit: TensorMap, id
using Test: @test

mps = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
before = dense(mps)
untouched = copy(mps[2])
site = 4  # The right bond of site 4 connects sites 4 and 1.
V = right_virtualspace(mps, site)
R = TensorMap(ComplexF64[2 0; 0 0.5], V, V)
L = inv(R)
@test R * L ≈ id(ComplexF64, V)

# Absorb R into site 4's right leg and L into site 1's left leg.
@test insert_gauge!(mps, site, R, L) === mps
@test mps[2] == untouched
@test assertconsistency(mps) === nothing
@test dense(mps) ≈ before
@show right_virtualspace(mps, site)
# R * L = I preserves the state. Arbitrary matrices need not preserve it.
nothing
