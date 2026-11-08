# Run from PMPSLab: julia --project=. examples/optimization/12_configuration_json.jl
using PMPSLab: GaugeInitialization, SplitEnvironmentPolicy, NuclearNorm, CostTolerance
using PMPSLab
import ExperimentLab, LatticeIteratorLab, TensorLinearSolveLab
using Test: @test

# Each package names its own configuration types. PMPSLab composes those
# registries, so applications do not need to repeat the registrations.
algorithm = ALSSplit(
    initialization=GaugeInitialization(),
    sweep=LatticeIteratorLab.CircularSweep(max_sweeps=4),
    solver=TensorLinearSolveLab.CholeskySolve(),
    environments=SplitEnvironmentPolicy(mixed=PMPSLab.TruncatedTransferPolicy()),
    regularization=NuclearNorm(),
    stopping=CostTolerance(atol=1e-8, rtol=1e-6),
)

directory = mktempdir()
path = joinpath(directory, "algorithm.json")
ExperimentLab.writeconfig(path, algorithm)
println(read(path, String))

# The root, sweep, solver, initialization, policies, regularizer and stopping
# rule each have a stable type name, even when they have no parameters.
registry = ExperimentLab.config_registry(typeof(algorithm))
document = ExperimentLab.config_record(registry, algorithm)
@show document["type"]
@show document["parameters"]["solver"]
@show document["parameters"]["regularization"]["parameters"]["schedule"]

# Supplying the concrete type also identifies third-party field types, such as
# MatrixAlgebraKit truncation policies, for their existing conversion hooks.
restored = ExperimentLab.readconfig(path, typeof(algorithm))
@test isequal(restored, algorithm)

# If all fields can be inferred, the registry alone is sufficient.
policy_path = joinpath(directory, "environments.json")
environments = SplitEnvironmentPolicy()
ExperimentLab.writeconfig(policy_path, environments)
@test ExperimentLab.readconfig(policy_path, registry) == environments

# record_algorithm! uses the same JSON representation automatically.
store = ExperimentLab.DataStore(joinpath(directory, "experiment"))
ref = ExperimentLab.record_algorithm!(store, algorithm)
@test ExperimentLab.info(store, ref)["config"] == document
@show path
nothing
