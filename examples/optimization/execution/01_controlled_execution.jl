# julia --project=. -i examples/optimization/execution/01_controlled_execution.jl
using PMPSLab
import AlgorithmsInterface as AI
import LatticeIteratorLab, TensorLinearSolveLab

target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=2),
    solver=TensorLinearSolveLab.CholeskySolve())
problem = PMPSLab.SplitProblem(target)
state = AI.initialize_state(problem, algorithm)

# The calling loop owns control. Return to the REPL after the second site.
while !AI.is_finished!(problem, algorithm, state)
    AI.increment!(state)
    AI.step!(problem, algorithm, state)
    state.progress.completed_updates == 2 && break
end
@assert state.progress.completed_updates == 2
@show state.progress

# Continue by running the same loop without the break, or resume a snapshot:
# result = AI.solve(problem, algorithm; snapshot=PMPSLab.snapshot(state))
# AI.solve!(problem, algorithm, state) instead resets the iteration and stopping history.
nothing
