# Run from PMPSLab: julia --project=. examples/optimization/07_exact_error.jl
using PMPSLab: SplitEnvironmentPolicy, CostConvergence
using PMPSLab
import LatticeIteratorLab, TensorLinearSolveLab
using TensorKit: truncrank
using Random: seed!
using Test: @test

seed!(462)
target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
initial = rand(PeriodicMPS, ComplexF64, 4, 2, 1)

# Choose transfer policies separately for the normal equation and the mixed
# right-hand side. The policy determines the resulting transfer representation.
mixed_policy = PMPSLab.TruncatedTransferPolicy(trunc=truncrank(1))
mixed_transfer = PMPSLab.transfer_at(initial, target, 1, mixed_policy)
@test mixed_transfer isa PMPSLab.TruncatedTransferMatrix

algorithm = ALSSplit(
    sweep=LatticeIteratorLab.CircularSweep(max_sweeps=2),
    solver=TensorLinearSolveLab.LUSolve(),
    environments=SplitEnvironmentPolicy(mixed=mixed_policy),
    error_measure=:exact,
    stopping=CostConvergence(atol=1e-10, rtol=1e-8),
)
result = split(target, initial, algorithm)
approximation = result.mps

# Site updates report estimates from the working environments.
# Each completed sweep additionally contracts the true squared distance.
# Because error_measure=:exact, the stopping criterion uses exact_error.
println("Final local estimate: ", result.progress.local_cost)
println("Final exact error: ", result.progress.exact_error)
@test result.progress.exact_error ≈ PMPSLab.squared_distance(target, approximation)
@test !isapprox(result.progress.local_cost, result.progress.exact_error; rtol=1e-5)

# The default error_measure=:local avoids the extra global contractions.
# In that mode exact_error is nothing, and stopping uses the local estimate.
nothing
