# Run 01 first. This file also works in a fresh Julia session.
include(joinpath(@__DIR__, "setup.jl"))
identity = JSON.parsefile(joinpath(workspace, "last-run.json"))
trial = getrun(client, identity["run_id"])
artifact_dir = chop(trial.info.artifact_uri; head=7, tail=0)

# Remeasure an existing snapshot: a standard child run and an input URI tag.
snapshot_path = "states/step-2.jld2"
measurement = createrun(client, identity["experiment_id"]; run_name="spectrum at step 2",
    start_time=round(Int, 1000time()),
    tags=Dict("mlflow.parentRunId" => trial.info.run_id,
        "kind" => "measurement", "input_step" => "2",
        "input_snapshot" => "runs:/$(trial.info.run_id)/$snapshot_path"))
measurement_dir = chop(measurement.info.artifact_uri; head=7, tail=0)
mkpath(measurement_dir)
logparam(client, measurement, "method", "eigvals")

input_tensor = JLD2.load(joinpath(artifact_dir, snapshot_path), "tensor")
open(joinpath(measurement_dir, "measurement.jsonl"), "w") do io
    with_logger(FormatLogger(LoggingFormats.JSON(; recursive=true), io)) do
        @info "Measurement start"
        levels = sort(real.(eigvals(convert(Array, input_tensor))))
        result = (; levels)
        jldsave(joinpath(measurement_dir, "result.jld2"); result)
        write(joinpath(measurement_dir, "result.json"), JSON.json(result))
        logmetric(client, measurement, "lowest_level", first(levels);
            step=2, timestamp=round(Int, 1000time()))
        @info "Measurement done" levels
    end
end
updaterun(client, measurement; status=RunStatus.FINISHED, end_time=round(Int, 1000time()))

# An independent copy can be measured asynchronously WITHOUT saving its tensor.
memory_measurement = createrun(client, identity["experiment_id"];
    run_name="memory-only norm", start_time=round(Int, 1000time()),
    tags=Dict("mlflow.parentRunId" => trial.info.run_id,
        "kind" => "measurement", "input_step" => "3", "input_storage" => "memory-only"))
memory_dir = chop(memory_measurement.info.artifact_uri; head=7, tail=0)
mkpath(memory_dir)
logparam(client, memory_measurement, "method", "norm")
live_tensor = copy(input_tensor)
measurement_input = copy(live_tensor)
ready = Channel{Nothing}(1)
measurement_task = @async begin
    take!(ready)
    open(joinpath(memory_dir, "measurement.jsonl"), "w") do io
        with_logger(FormatLogger(LoggingFormats.JSON(; recursive=true), io)) do
            @info "Memory measurement start"
            result = (value=norm(measurement_input),)
            jldsave(joinpath(memory_dir, "result.jld2"); result)
            write(joinpath(memory_dir, "result.json"), JSON.json(result))
            @info "Memory measurement done" result.value
        end
    end
    updaterun(client, memory_measurement;
        status=RunStatus.FINISHED, end_time=round(Int, 1000time()))
end
rmul!(live_tensor, 0) # The algorithm can modify its own tensor independently.
put!(ready, nothing)
fetch(measurement_task) # Wait explicitly; this example uses one Julia thread.
@show measurement.info.run_id memory_measurement.info.run_id
nothing
