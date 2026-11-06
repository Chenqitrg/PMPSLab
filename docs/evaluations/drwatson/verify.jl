include(joinpath(@__DIR__, "02_reopen_and_collect.jl"))
using Test

@testset "DrWatson local scientific workflow" begin
    @test record["status"] == "completed"
    @test record["settings"] == (solver=(tolerance=1e-10, maxiter=80), max_steps=2)
    @test haskey(record, "gitcommit")
    @test tensor isa TensorMap
    @test tensor == TensorMap(ComplexF64[0.5 0; 0 1.5], ComplexSpace(2), ComplexSpace(2))
    @test result isa NamedTuple
    @test result.levels ≈ [0.5, 1.5]
    @test measurement["parent_run"] == ids["run_id"]
    @test measurement["input_snapshot"] == record["checkpoint"]
    @test length(readdir(joinpath(run_dir, "snapshots"))) == 2
    @test Set(readdir(measurement_dir)) == Set(["result.jld2", "result.json", "measurement.jsonl"])
    @test first(events)["msg"] == "Algorithm start"
    @test cost_history == [(step=1, cost=0.1), (step=2, cost=0.1^2)]
    @test JSON.parse(readline(joinpath(measurement_dir, "measurement.jsonl")))["msg"] == "Measurement start"
    row = only(eachrow(filter(:run_id => ==(ids["run_id"]), table)))
    @test row.final_cost ≈ 0.01
    @test joinpath(workspace, row.run_path) == run_dir
end

# Verify the upstream caching behavior: identical names reuse previous output.
# This is optional caching, not the default used by our independent runs above.
cache_dir = mktempdir(workspace; prefix="cache-", cleanup=false)
cached, cache_file = produce_or_load((chi=16,), cache_dir; tag=false, verbose=false) do config
    Dict("value" => 1)
end
reused, reused_file = produce_or_load((chi=16,), cache_dir; tag=false, verbose=false) do config
    error("An existing result should bypass this changed function")
end
@testset "Optional cache semantics" begin
    @test cached == reused
    @test cache_file == reused_file
end
