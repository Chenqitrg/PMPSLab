# Run 01 and 02 first. Reopen using only a client and the saved run ID.
include(joinpath(@__DIR__, "setup.jl"))
identity = JSON.parsefile(joinpath(workspace, "last-run.json"))
trial = getrun(client, identity["run_id"])
artifact_dir = chop(trial.info.artifact_uri; head=7, tail=0)

settings = JLD2.load(joinpath(artifact_dir, "config.jld2"), "settings")
snapshot = JLD2.load(joinpath(artifact_dir, "states", "step-2.jld2"), "tensor")
history, next_page = getmetrichistory(client, trial, "cost")
children, next_children = searchruns(client;
    experiment_ids=[identity["experiment_id"]],
    filter="tags.mlflow.parentRunId = '$(trial.info.run_id)'")
root_uri, artifacts, next_artifacts = listartifacts(client, trial)

@show settings typeof(snapshot)
@show [(m.step, m.value) for m in sort(history; by=m -> m.step)]
@show [(child.info.run_name, child.info.status) for child in children]
@show [file.path for file in artifacts]
println(read(joinpath(artifact_dir, "run.jsonl"), String))

# Load the typed measurement result; JSON remains available for the web frontend.
spectrum_run = only(filter(child -> child.info.run_name == "spectrum at step 2", children))
spectrum_dir = chop(spectrum_run.info.artifact_uri; head=7, tail=0)
result = JLD2.load(joinpath(spectrum_dir, "result.jld2"), "result")
@show result.levels # [0.5, 1.5]
nothing
