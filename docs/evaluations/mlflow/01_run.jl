# A local scientific run, using existing packages directly.
include(joinpath(@__DIR__, "setup.jl"))

# Client and server must see this same directory (local disk or shared storage).
# MLflow owns run identity; JLD2 writes directly to the run's artifact directory.
experiment_id = createexperiment(client, "tensor-review-$(uuid4())";
    artifact_location="file://$(joinpath(workspace, "artifacts"))")
trial = createrun(client, experiment_id; run_name="two steps",
    start_time=round(Int, 1000time()))
run_id = trial.info.run_id
@assert startswith(trial.info.artifact_uri, "file://")
artifact_dir = chop(trial.info.artifact_uri; head=7, tail=0)
mkpath(joinpath(artifact_dir, "states"))

# Ordinary Julia configuration, with a readable copy and a typed Julia copy.
settings = (solver=(tolerance=1e-10, maxiter=80), max_steps=2)
logparam(client, trial, "solver.tolerance", string(settings.solver.tolerance))
logparam(client, trial, "solver.maxiter", string(settings.solver.maxiter))
logparam(client, trial, "max_steps", string(settings.max_steps))
write(joinpath(artifact_dir, "config.json"), JSON.json(settings))
jldsave(joinpath(artifact_dir, "config.jld2"); settings)

# Standard Julia logging. The enclosing run directory supplies its identity.
# Logs can be opened while the algorithm is running; flush each event to disk.
tensor = TensorMap(ComplexF64[1 0; 0 3], ComplexSpace(2), ComplexSpace(2))
open(joinpath(artifact_dir, "run.jsonl"), "w") do io
    logger = TeeLogger(ConsoleLogger(),
        FormatLogger(LoggingFormats.JSON(; recursive=true, nest_kwargs=false), io))
    with_logger(logger) do
        @info "Algorithm start" run_id
        for step in 1:settings.max_steps
            # Explicit milliseconds avoid MLFlowClient 0.10.0's default timestamp bug.
            logmetric(client, trial, "cost", 0.1^step;
                step, timestamp=round(Int, 1000time()))
            jldsave(joinpath(artifact_dir, "states", "step-$step.jld2"); tensor, step)
            @info "Step completed" step cost=0.1^step
            rmul!(tensor, 0.5)
        end
        @info "Algorithm done"
    end
end

# A checkpoint is a reference to an existing snapshot, not a second tensor file.
setruntag(client, trial, "checkpoint", "states/step-2.jld2")
updaterun(client, trial; status=RunStatus.FINISHED, end_time=round(Int, 1000time()))

# Keep only enough information to reopen from a new Julia session.
write(joinpath(workspace, "last-run.json"), JSON.json((; experiment_id, run_id)))
@show run_id artifact_dir
nothing
