# Run in an experiment environment with PMPSLab, DrWatson and JLD2.
using PMPSLab, DrWatson, JLD2, Random
import LatticeIteratorLab, TensorLinearSolveLab

function compute_split(settings::NamedTuple)::Dict{String,Any}
    Random.seed!(settings.seed)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=settings.sweeps),
        solver=TensorLinearSolveLab.CholeskySolve())
    result = split(target, algorithm)
    return Dict("settings" => settings, "mps" => result.mps, "progress" => result.progress)
end

settings = (seed=1234, sweeps=2, revision="demo-v1")
output = mktempdir()
data, file = produce_or_load(compute_split, settings, output; tag=false)
reused, reused_file = produce_or_load(compute_split, settings, output; tag=false)
@assert file == reused_file
@assert dense(data["mps"]) == dense(reused["mps"])
@show file data["progress"]
# For real experiments, identify the target and code/environment in the cache key.
# Git tagging records provenance; it does not automatically invalidate old results.
nothing
