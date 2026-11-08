# Run from PMPSLab: julia --project=. examples/gauge/07_configuration_tree.jl
using ExperimentLab: @config
import PMPSLab
using TensorKit: MatrixAlgebraKit
using Test: @test

config = @config """
PMPSLab.ProjectorGaugeFix
├─ tol: 1e-13
└─ trunc: MatrixAlgebraKit.truncrank(8)
"""

# Ordinary display automatically includes all fields in an editable tree.
display(config)
@test config.tol == 1e-13
nothing
