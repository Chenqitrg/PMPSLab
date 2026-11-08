# Run from PMPSLab: julia --project=. examples/optimization/06_existing_approximation.jl
using PMPSLab: SVDInitialization
using PMPSLab
import LatticeIteratorLab, TensorLinearSolveLab
using TensorKit: truncrank, domain
using Random: seed!
using Test: @test

seed!(1234)
target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
initial = initial_approximation(target, SVDInitialization(trunc=truncrank(1)))
before = deepcopy(initial)
algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=2), solver=TensorLinearSolveLab.LUSolve())

# The supplied initial state fixes the optimization spaces. No initialization
# strategy runs here, and the site count is inferred from the actual MPS.
optimized = split(target, initial, algorithm).mps
@test optimized !== initial
@test dense(initial) == dense(before)
@test all(domain(optimized[i]) == domain(initial[i]) for i in eachindex(initial))

# Opt out of the full initial copy to update the caller's MPS container.
# Local solves and transfer caches still require working memory.
inplace_result = split(target, initial, algorithm; copy_initial=false)
inplace = inplace_result.mps
@test inplace === initial
@test dense(inplace) ≈ dense(optimized)
@test inplace_result.progress.completed_sweeps == 2
println("Copied result preserves the input; copy_initial=false returns the input MPS itself.")

# The controlled interface has the same ownership option:
# run = PMPSLab.initialize_split(target, initial, algorithm; copy_initial=false)
nothing
