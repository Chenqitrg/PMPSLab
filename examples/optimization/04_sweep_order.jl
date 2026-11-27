# Run from PMPSLab: julia --project=. examples/optimization/04_sweep_order.jl
using PMPSLab
import LatticeIteratorLab, TensorLinearSolveLab
import AlgorithmsInterface as AI
using Test: @test
using Random: seed!

seed!(1234)

circular = collect(LatticeIteratorLab.sweep_states(LatticeIteratorLab.CircularSweep(max_sweeps=2), 4))
zigzag = collect(LatticeIteratorLab.sweep_states(LatticeIteratorLab.ZigzagSweep(max_sweeps=2), 4))

@test [state.site for state in circular] == [1, 2, 3, 4, 1, 2, 3, 4]
@test [state.site for state in zigzag] == [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2]
@test [state.completed_sweeps for state in circular] == [0, 0, 0, 0, 1, 1, 1, 1]
@test [state.completed_sweeps for state in zigzag] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

println("Circular site order: ", [state.site for state in circular])
println("Zigzag site order:   ", [state.site for state in zigzag])

# Pass the same policy to PMPSLab. It follows LatticeIteratorLab's order while
# solving tensors; LatticeIteratorLab itself knows nothing about MPS or linear solves.
target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
algorithm = ALSSplit(sweep=LatticeIteratorLab.ZigzagSweep(max_sweeps=2),
    solver=TensorLinearSolveLab.CholeskySolve())
state = PMPSLab.initialize_split(target, algorithm)
visited_sites = Int[]
problem = PMPSLab.SplitProblem(target)
while !AI.is_finished!(problem, algorithm, state)
    AI.increment!(state)
    AI.step!(problem, algorithm, state)
    push!(visited_sites, state.progress.site)
end
result = AI.finalize_state!(problem, algorithm, state)
@test visited_sites == [position.site for position in zigzag]
@test result.progress.completed_sweeps == 2
# A sweep state describes an upcoming update. completed_sweeps counts full
# sweeps before that update; 0 labels the first sweep. The scheme also records
# the direction used by the cache refresh. Constructing a sweep does no numerical work.
nothing
