# Run from PMPSLab: julia --project=. examples/optimization/01_simple_split.jl
using PMPSLab: SVDInitialization
using PMPSLab
import LatticeIteratorLab, TensorLinearSolveLab, ExperimentLab
using Random: seed!
using Test: @test

seed!(1234)
# 1. PMPSLab supplies the data: two sites, two physical legs each.
target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)

# 2. Independent packages supply the visit order and local linear solver.
# Constructing these settings does not run any tensor calculation.
sweep = LatticeIteratorLab.CircularSweep(max_sweeps=1)
solver = TensorLinearSolveLab.CholeskySolve()

# 3. PMPSLab combines those choices into the ALS algorithm.
algorithm = ALSSplit(
    initialization=SVDInitialization(),
    sweep=sweep,
    solver=solver,
)
# 4. the split algorithm's generator expresses ALS; ExperimentLab drives its site updates.
state = PMPSLab.initialize_split(target, algorithm)
session = ExperimentLab.start_run(PMPSLab.split_steps(state); threaded=false)
result = fetch(session)
# The blocking shortcut is equivalent: result = split(target, algorithm).
approximation = result.mps

# The same four physical legs are now carried by four single-leg sites.
@test length(target) == 2
@test length(approximation) == 4
@test dense(approximation) ≈ dense(target)
@test result.progress.completed_updates == 4
println("Target sites: ", length(target), "; approximation sites: ", length(approximation))
println("Site updates: ", result.progress.completed_updates)
# With no SVD truncation, initialization is already exact. Example 02 introduces
# compression, so the later ALS updates have a nonzero error to reduce.
nothing
