# Run 01 first; this file can run in a fresh Julia process.
include(joinpath(@__DIR__, "setup.jl"))
using DrWatson, JLD2, TensorKit, DataFrames
import JSON

ids = JSON.parsefile(joinpath(workspace, "last-run.json"))
run_dir = joinpath(workspace, "runs", ids["run_id"])
record = wload(joinpath(run_dir, "run.jld2"))
tensor = wload(joinpath(run_dir, record["checkpoint"]))["tensor"]
measurement_dir = joinpath(run_dir, "measurements", ids["measurement_id"])
measurement = wload(joinpath(measurement_dir, "result.jld2"))
result = measurement["result"]
events = JSON.parse.(readlines(joinpath(run_dir, "run.jsonl")))
cost_history = [(step=event["step"], cost=event["cost"])
    for event in events if haskey(event, "cost")]

# DrWatson collects small summary files into a DataFrame.
# It does not provide a tracking server or web UI.
table = collect_results(joinpath(workspace, "summaries"); verbose=false)
@show record["settings"] typeof(tensor) result.levels cost_history
display(table)
nothing
