using LatticeIteratorLab
using TensorLinearSolveLab
# Run from PMPSLab: julia --project=. examples/optimization/extras/configuration_tree.jl
using ExperimentLab: @config
import PMPSLab
using Test: @test

sweep = @config """
LatticeIteratorLab.CircularSweep
  max_sweeps: 4
"""
config = @config """
PMPSLab.ALSSplit
├─ initialization: PMPSLab.SVDInitialization()
├─ solver: TensorLinearSolveLab.CholeskySolve()
└─ sweep: sweep
"""
@test config.sweep === sweep
display(config)
@test config.solver isa TensorLinearSolveLab.CholeskySolve
# This constructs settings only; it does not run an MPS optimization.
nothing
