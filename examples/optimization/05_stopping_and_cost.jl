# Run from PMPSLab: julia --project=. examples/optimization/05_stopping_and_cost.jl
using PMPSLab: SVDInitialization, NoStopping, CostConvergence
using PMPSLab
import LatticeIteratorLab, TensorLinearSolveLab
using Random: seed!
using Test: @test

seed!(1234)
target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
algorithm = ALSSplit(
    initialization=SVDInitialization(),
    sweep=LatticeIteratorLab.CircularSweep(max_sweeps=6),
    solver=TensorLinearSolveLab.CholeskySolve(),
    error_measure=:exact,
    stopping=CostConvergence(atol=1e-10, rtol=1e-8, min_sweeps=2, consecutive_sweeps=1),
)
result = split(target, algorithm)
approximation = result.mps

# An already exact initial guess gives a simple, predictable stopping example.
# Stops are checked after complete sweeps, not after every site update.
@test result.progress.converged
@test result.progress.completed_updates == 8  # Two sweeps rather than the allowed six.
@test isapprox(result.progress.exact_error, PMPSLab.squared_distance(target, approximation); atol=1e-9)
println("Completed sweeps: ", result.progress.completed_sweeps)
println("Final exact error: ", result.progress.exact_error)
# CostConvergence detects a plateau. It does not guarantee a small absolute
# error or a global optimum. Without a stopping criterion, NoStopping() is used.
nothing
