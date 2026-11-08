using PMPSLab: find_projectors
# Run from PMPSLab: julia --project=. examples/gauge/04_gaugefix.jl
# Gauge fixing combines projector construction and absorption into the MPS.
using PMPSLab: ProjectorGaugeFix, find_projectors, gaugefix!
using PMPSLab: PeriodicMPS, dense, insert_gauge!, assertconsistency
using LinearAlgebra: norm
using Test: @test

original = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
before = dense(original)
algorithm = ProjectorGaugeFix(; tol=1e-12, maxiter=100)

# For these structurally full-rank tensors, spell out the projector workflow.
manual = deepcopy(original)
PR, PL = find_projectors(manual, algorithm)
for i in eachindex(manual)
    insert_gauge!(manual, i, PR[i], PL[i])
end

fixed = deepcopy(original)
@test gaugefix!(fixed, algorithm) === fixed
@test assertconsistency(fixed) === nothing
@test dense(fixed) ≈ before rtol=1e-9
@test dense(manual) ≈ dense(fixed) rtol=1e-9
@show norm(dense(fixed) - before) / norm(before)
# gaugefix! changes the representation in place; it does not normalize the state.
# It also performs structural cleanup first when necessary (example 06).
nothing
