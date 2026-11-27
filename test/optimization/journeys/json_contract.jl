# Read application summaries without importing tensor types or a persistence API.
function check_cdl_json(root::AbstractString; sweeps::Int=50, sites::Int=8)::Nothing
    last_step = sweeps * sites
    for label in ("als", "nnr", "resumed")
        record = JSON.parsefile(joinpath(root, "$label.json"))
        @test record["label"] == label
        offset = label == "resumed" ? last_step ÷ 2 : 0
        @test record["checkpoint_step"] == offset
        @test record["parent_run"] == (label == "resumed" ? "nnr" : nothing)
        rows = record["events"]
        @test [row["completed_updates"] for row in rows] == collect(offset+1:last_step)
        @test all(row["duration"] >= 0 && isfinite(row["duration"]) for row in rows)
        @test all(row["site"] == mod1(row["completed_updates"], sites) for row in rows)
        @test all(row["sweep"] == cld(row["completed_updates"], sites) for row in rows)
        @test all(row["completed_sweeps"] == row["completed_updates"] ÷ sites for row in rows)
        @test all(row["cost_scale"] > 0 && isfinite(row["local_cost"]) for row in rows)
        @test all(label == "als" ? isnothing(row["penalty"]) : row["penalty"] > 0 for row in rows)
        data = record["measurement"]
        @test data["relative_squared_error"] < 1e-10
        if label == "als"
            @test data["max_second_ratio"] > 0.1
            @test data["rank1_relative_change"] > 1e-4
        else
            @test all(==(1), data["effective_pair_ranks"])
            @test data["max_second_ratio"] < 1e-6
            @test data["rank1_relative_change"] < 1e-16
        end
    end
    return nothing
end
