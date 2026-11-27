# Run in an experiment environment with PMPSLab and JSON.
using PMPSLab, JSON
import LatticeIteratorLab, TensorLinearSolveLab
using Test: @test

# The application chooses its portable parameter schema. It does not encode Julia types.
settings = Dict("sweeps" => 4, "initial_penalty" => 0.01, "decay" => 0.95)
file = joinpath(mktempdir(), "settings.json")
write(file, JSON.json(settings))
loaded = JSON.parsefile(file)
@test loaded == settings

algorithm = ALSSplit(
    sweep=LatticeIteratorLab.CircularSweep(max_sweeps=loaded["sweeps"]),
    solver=TensorLinearSolveLab.CholeskySolve(),
    regularization=PMPSLab.NuclearNorm(schedule=PMPSLab.GeometricDecay(
        initial=loaded["initial_penalty"], factor=loaded["decay"])),
)
@test algorithm.regularization.schedule.factor == 0.95
# Use JLD2 when the complete Julia algorithm, including third-party criteria, is needed.
@show file algorithm
nothing
