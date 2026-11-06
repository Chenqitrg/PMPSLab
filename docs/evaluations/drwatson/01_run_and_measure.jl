include(joinpath(@__DIR__, "setup.jl"))
using DrWatson, JLD2, TensorKit, LinearAlgebra, Logging, LoggingExtras, UUIDs
import JSON, LoggingFormats

# IDs, directories, and input paths below are application conventions.
# DrWatson does not create a run object or enforce these relationships.
run_id = string(uuid4())
run_dir = joinpath(workspace, "runs", run_id)
mkpath(run_dir)
settings = (solver=(tolerance=1e-10, maxiter=80), max_steps=2)
metadata = Dict("run_id" => run_id, "settings" => settings, "status" => "running")
tagsave(joinpath(run_dir, "run.jld2"), metadata; gitpath=repository, warn=false)
write(joinpath(run_dir, "config.json"), JSON.json(settings))

tensor = TensorMap(ComplexF64[1 0; 0 3], ComplexSpace(2), ComplexSpace(2))
snapshots = String[]
costs = Float64[]
try
    open(joinpath(run_dir, "run.jsonl"), "w") do io
        with_logger(FormatLogger(LoggingFormats.JSON(; recursive=true, nest_kwargs=false), io)) do
            @info "Algorithm start" run_id
            for step in 1:settings.max_steps
                cost = 0.1^step
                push!(costs, cost)
                snapshot = joinpath("snapshots", "step-$step.jld2")
                wsave(joinpath(run_dir, snapshot), Dict("tensor" => tensor, "step" => step))
                push!(snapshots, snapshot)
                @info "Step completed" step cost
                rmul!(tensor, 0.5)
            end
            @info "Algorithm done"
        end
    end
    metadata["status"] = "completed"
catch exception
    metadata["status"] = "failed"
    metadata["error"] = sprint(showerror, exception)
    rethrow()
finally
    # Reuse the Git metadata captured at the start, rather than tagging again.
    recorded = wload(joinpath(run_dir, "run.jld2"))
    merge!(recorded, metadata)
    recorded["checkpoint"] = isempty(snapshots) ? nothing : last(snapshots)
    wsave(joinpath(run_dir, "run.jld2"), recorded)
end

# A measurement has its own folder, configuration, log, and input reference.
measurement_id = string(uuid4())
measurement_dir = joinpath(run_dir, "measurements", measurement_id)
mkpath(measurement_dir)
input_path = last(snapshots)
input_tensor = wload(joinpath(run_dir, input_path))["tensor"]
open(joinpath(measurement_dir, "measurement.jsonl"), "w") do io
    with_logger(FormatLogger(LoggingFormats.JSON(; recursive=true), io)) do
        @info "Measurement start"
        levels = sort(real.(eigvals(convert(Array, input_tensor))))
        result = (; levels)
        tagsave(joinpath(measurement_dir, "result.jld2"),
            Dict("settings" => (method="eigvals",), "result" => result,
                "parent_run" => run_id, "input_snapshot" => input_path);
            gitpath=repository, warn=false)
        write(joinpath(measurement_dir, "result.json"), JSON.json(result))
        @info "Measurement done" levels
    end
end

# Keep only SMALL records here: collecting results must not load every tensor.
summary_file = joinpath(workspace, "summaries", "$run_id.jld2")
wsave(summary_file, Dict("run_id" => run_id, "tolerance" => settings.solver.tolerance,
    "final_cost" => last(costs), "run_path" => relpath(run_dir, workspace)))
write(joinpath(workspace, "last-run.json"), JSON.json((; run_id, measurement_id)))
@show run_dir summary_file
nothing
