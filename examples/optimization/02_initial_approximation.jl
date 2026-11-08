# Run from PMPSLab: julia --project=. examples/optimization/02_initial_approximation.jl
using PMPSLab: SVDInitialization
using PMPSLab
using TensorKit: truncrank
using Random: seed!
using Test: @test

seed!(1234)
target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
exact = initial_approximation(target, SVDInitialization())
compressed = initial_approximation(target, SVDInitialization(trunc=truncrank(1)))

# SVD splits each target tensor in two. Truncation limits the new bond inside
# each pair; it does not change every existing virtual bond to dimension one.
@test length(exact) == length(compressed) == 4
@test dense(exact) ≈ dense(target)
initial_error = PMPSLab.squared_distance(target, compressed)
@test initial_error > 0
println("Exact SVD squared error: ", PMPSLab.squared_distance(target, exact))
println("Compressed SVD squared error: ", initial_error)
# This only constructs an initial guess. No ALS sweep has run yet.
nothing
