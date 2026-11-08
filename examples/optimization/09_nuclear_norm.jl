# Run from PMPSLab: julia --project=. examples/optimization/09_nuclear_norm.jl
using PMPSLab
using ExperimentLab: @config
import ExperimentLab, LatticeIteratorLab, TensorLinearSolveLab
using Random: seed!
using Test: @test

seed!(710)
target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
target[1] *= 11 # Deliberately not normalized.
original = deepcopy(target)

# The NNR parameters form one subtree; the solver and sweep are reused.
algorithm = @config """
PMPSLab.ALSSplit
├─ sweep: LatticeIteratorLab.CircularSweep
│  └─ max_sweeps: 8
├─ solver: TensorLinearSolveLab.CholeskySolve()
├─ regularization: PMPSLab.NuclearNorm
│  └─ schedule: PMPSLab.GeometricDecay
│     ├─ initial: 1.0e-5
│     ├─ factor: 0.8
│     └─ minimum: 1.0e-7
└─ error_measure: :exact
"""
display(algorithm)

# An external registry gives JSON nodes stable names without adding fields to
# these configurations. The root and every nested Lab configuration are named.
ExperimentLab.writeconfig(stdout, algorithm)
# {"type":"PMPSLab.ALSSplit","parameters":{"sweep":{...}, ...}}

result = split(target, algorithm)

# The solve uses normalized equations, but the result keeps the original units.
# Both the ADMM penalty and the soft threshold equal progress.penalty (xi).
# The nuclear-norm weight is xi^2, following the paper.
@show result.progress.penalty
@show result.progress.exact_error # ||target - result.mps||^2 / ||target||^2
physical_error = result.progress.exact_error * result.progress.cost_scale
@test physical_error ≈ PMPSLab.squared_distance(target, result.mps) atol=1e-10
@test all(target[i] == original[i] for i in eachindex(target))

# State/callback tensors are also in original units. No extra target copy is
# needed for normalization. NoRegularization() restores ordinary ALS behavior.
# With ZigzagSweep, one decay occurs after a complete round trip.
nothing
