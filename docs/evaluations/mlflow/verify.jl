# Evaluation checks, deliberately separate from the REPL walkthrough.
include(joinpath(@__DIR__, "03_reopen.jl"))
using Test

@testset "Direct MLflow scientific workflow" begin
    @test trial.info.status == RunStatus.FINISHED
    @test trial.info.start_time > 1_700_000_000_000
    @test trial.info.end_time >= trial.info.start_time
    @test settings == (solver=(tolerance=1e-10, maxiter=80), max_steps=2)
    @test snapshot isa TensorMap
    @test snapshot == TensorMap(ComplexF64[0.5 0; 0 1.5], ComplexSpace(2), ComplexSpace(2))
    @test sort([(m.step, m.value) for m in history]) == [(1, 0.1), (2, 0.1^2)]
    @test all(m -> m.timestamp > 1_700_000_000_000, history)
    @test isnothing(next_page) || isempty(next_page)
    @test length(children) == 2
    @test all(child -> child.info.status == RunStatus.FINISHED, children)
    @test result isa NamedTuple
    @test result.levels ≈ [0.5, 1.5]
    @test JSON.parsefile(joinpath(spectrum_dir, "result.json"))["levels"] ≈ result.levels
    @test Set(file.path for file in artifacts) ==
        Set(["config.jld2", "config.json", "run.jsonl", "states"])
    log_entries = JSON.parse.(readlines(joinpath(artifact_dir, "run.jsonl")))
    @test first(log_entries)["msg"] == "Algorithm start"
    @test only(filter(entry -> get(entry, "step", 0) == 2, log_entries))["cost"] ≈ 0.01
    @test JSON.parse(readline(joinpath(spectrum_dir, "measurement.jsonl")))["msg"] == "Measurement start"
    @test length(readdir(joinpath(artifact_dir, "states"))) == 2
    @test only(filter(tag -> tag.key == "checkpoint", trial.data.tags)).value == "states/step-2.jld2"
    @test only(filter(tag -> tag.key == "input_snapshot", spectrum_run.data.tags)).value ==
        "runs:/$(trial.info.run_id)/states/step-2.jld2"

    memory_run = only(filter(child -> child.info.run_name == "memory-only norm", children))
    memory_dir = chop(memory_run.info.artifact_uri; head=7, tail=0)
    @test JLD2.load(joinpath(memory_dir, "result.jld2"), "result").value ≈ sqrt(2.5)
    @test Set(readdir(memory_dir)) == Set(["measurement.jsonl", "result.jld2", "result.json"])
end

# Probe the released Julia client's unmodified default behavior separately.
probe = createrun(client, identity["experiment_id"]; run_name="client timestamp probe",
    start_time=round(Int, 1000time()))
logmetric(client, probe, "default_timestamp", 1.0)
probe_history, _ = getmetrichistory(client, probe, "default_timestamp")
default_timestamp_is_milliseconds = only(probe_history).timestamp > 1_700_000_000_000
@show default_timestamp_is_milliseconds only(probe_history).timestamp
updaterun(client, probe; status=RunStatus.FINISHED, end_time=round(Int, 1000time()))

# A failed run can retain its diagnostics without producing a state/snapshot.
failed = createrun(client, identity["experiment_id"]; run_name="failure before first state",
    start_time=round(Int, 1000time()))
failed_dir = chop(failed.info.artifact_uri; head=7, tail=0)
mkpath(failed_dir)
try
    open(joinpath(failed_dir, "run.jsonl"), "w") do io
        with_logger(FormatLogger(LoggingFormats.JSON(; recursive=true), io)) do
            @info "Algorithm start"
            error("Deliberate evaluation failure")
        end
    end
catch exception
    write(joinpath(failed_dir, "error.txt"), sprint(showerror, exception))
    updaterun(client, failed; status=RunStatus.FAILED, end_time=round(Int, 1000time()))
end
@testset "Failure without a snapshot" begin
    @test getrun(client, failed.info.run_id).info.status == RunStatus.FAILED
    @test JSON.parse(readline(joinpath(failed_dir, "run.jsonl")))["msg"] == "Algorithm start"
    @test occursin("Deliberate evaluation failure", read(joinpath(failed_dir, "error.txt"), String))
    @test Set(readdir(failed_dir)) == Set(["run.jsonl", "error.txt"])
end

# Small files also work through the HTTP artifact proxy, with no shared disk.
proxy_experiment = createexperiment(client, "proxy-review-$(uuid4())")
proxy_run = createrun(client, proxy_experiment; run_name="small remote artifact",
    start_time=round(Int, 1000time()))
@assert startswith(proxy_run.info.artifact_uri, "mlflow-artifacts:/")
proxy_path = chopprefix(proxy_run.info.artifact_uri, "mlflow-artifacts:/") * "/result.json"
payload = Vector{UInt8}(codeunits(JSON.json((levels=[1.0, 3.0],))))
@testset "HTTP artifact proxy" begin
    @test uploadartifact(client, proxy_path, payload)
    @test downloadartifact(client, proxy_path) == payload
    _, proxy_files, _ = listartifacts(client, proxy_run)
    @test only(proxy_files).path == "result.json"
end
updaterun(client, proxy_run; status=RunStatus.FINISHED, end_time=round(Int, 1000time()))
