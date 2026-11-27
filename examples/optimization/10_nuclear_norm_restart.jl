# julia --project=. examples/optimization/10_nuclear_norm_restart.jl
using PMPSLab
import AlgorithmsInterface as AI
import LatticeIteratorLab, TensorLinearSolveLab
using Test: @test

target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=2),
    solver=TensorLinearSolveLab.CholeskySolve(), regularization=PMPSLab.NuclearNorm())
problem = PMPSLab.SplitProblem(target)
state = AI.initialize_state(problem, algorithm)
for _ in 1:3
    @test !AI.is_finished!(problem, algorithm, state)
    AI.increment!(state)
    AI.step!(problem, algorithm, state)
end

# Captures the MPS, convergence history, and NNR auxiliary/dual tensors.
saved = PMPSLab.snapshot(state)
recovered = AI.solve(problem, algorithm; snapshot=saved)
expected = AI.solve(problem, algorithm)
@test dense(recovered.mps) ≈ dense(expected.mps)
@test recovered.progress.penalty == expected.progress.penalty
@test saved.progress.completed_updates == 3
# An experiment script can persist saved with DrWatson.wsave or JLD2.jldsave.
nothing
