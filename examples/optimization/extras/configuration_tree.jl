# Ordinary constructors work with the pinned A.I. interface.
# Editable tree input/display will follow the upstream tree PRs.
using PMPSLab
import AlgorithmsInterface as AI
import LatticeIteratorLab, TensorLinearSolveLab

algorithm = ALSSplit(
    sweep=LatticeIteratorLab.CircularSweep(max_sweeps=4),
    solver=TensorLinearSolveLab.CholeskySolve(),
    stopping_criterion=AI.StopWhenAny(
        PMPSLab.CostConvergence(atol=1e-10), AI.StopAfterIteration(12)),
)
display(algorithm)
nothing
