# Frontend contract: only JSON/JSONL reads; no store API or native tensor payloads.
using ExperimentLab: JSON

function check_cdl_json(root; sweeps=50, sites=8)
    index = JSON.parsefile(joinpath(root, "index.json"))
    executions = Dict(run["label"] => run for run in values(index["executions"]))
    last_step = sweeps * sites
    for label in ("als", "nnr", "resumed")
        run = executions[label]
        @test run["status"] == "completed"
        @test haskey(index["algorithms"], run["parent"]["id"])
        config = index["algorithms"][run["parent"]["id"]]["config"]
        @test config["type"] == "PMPSLab.ALSSplit"
        parameters = config["parameters"]
        @test parameters["sweep"]["type"] == "LatticeIteratorLab.CircularSweep"
        @test parameters["solver"]["type"] == "TensorLinearSolveLab.KrylovSolve"
        @test parameters["initialization"]["type"] == "PMPSLab.SVDInitialization"
        @test parameters["stopping"]["type"] == "PMPSLab.NoStopping"
        @test parameters["environments"]["parameters"]["normal"]["type"] ==
            "PMPSLab.DenseTransferPolicy"
        regularization = parameters["regularization"]
        @test regularization["type"] == (label == "als" ? "PMPSLab.NoRegularization" : "PMPSLab.NuclearNorm")
        if label != "als"
            @test regularization["parameters"]["schedule"]["type"] == "PMPSLab.GeometricDecay"
        end
        @test haskey(index["states"], run["initial_state"]["id"])
        rows = JSON.parse.(readlines(joinpath(root, run["log_path"])))
        @test all(row["level"] != "warn" && row["level"] != "error" for row in rows)
        offset = label == "resumed" ? last_step ÷ 2 : 0
        @test [row["data"]["completed_updates"] for row in rows] == collect(offset+1:last_step)
        @test all(row["owner"]["id"] == run["id"] && row["store_id"] == index["id"] for row in rows)
        @test all(row["data"]["duration"] >= 0 for row in rows)
        @test all(row["data"]["site"] == mod1(row["data"]["completed_updates"], sites) for row in rows)
        @test all(row["data"]["sweep"] == cld(row["data"]["completed_updates"], sites) for row in rows)
        @test all(row["data"]["cost_scale"] > 0 for row in rows)
        @test all(isfinite(row["data"]["local_cost"]) for row in rows)
        @test all(isfinite(row["data"]["duration"]) for row in rows)
        @test all(row["data"]["completed_sweeps"] == row["data"]["completed_updates"] ÷ sites for row in rows)
        if label == "als"
            @test all(isnothing(row["data"]["penalty"]) for row in rows)
        else
            @test all(row["data"]["penalty"] > 0 for row in rows)
        end

        attempts = filter(collect(values(index["measurements"]))) do attempt
            state = index["states"][attempt["parent"]["id"]]
            state["parent"]["id"] == run["id"] && state["step"] == last_step
        end
        attempt = only(attempts)
        @test attempt["status"] == "completed"
        @test attempt["config"]["type"] == "CDLJourney.Diagnostics"
        @test attempt["config"]["parameters"]["rank_rtol"] == 1e-6
        payload = JSON.parsefile(joinpath(root, attempt["result"]))
        @test payload["attempt"] == attempt["id"]
        measurement_logs = JSON.parse.(readlines(joinpath(root, attempt["log_path"])))
        @test !isempty(measurement_logs)
        @test all(row["owner"]["kind"] == "measurement" && row["owner"]["id"] == attempt["id"]
                  for row in measurement_logs)
        data = payload["data"]
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
    @test executions["resumed"]["parent"] == executions["nnr"]["parent"]
    @test executions["resumed"]["id"] != executions["nnr"]["id"]
    origin = index["states"][executions["resumed"]["initial_state"]["id"]]
    @test origin["parent"]["id"] == executions["nnr"]["id"]
    @test origin["step"] == last_step ÷ 2
    @test length(index["snapshots"]) == 3
    for saved in values(index["snapshots"])
        @test haskey(index["states"], saved["parent"]["id"])
        @test saved["stored"] && saved["format"] == "jld2-v1"
        @test endswith(saved["payload"], ".jld2")
    end
    paused = JSON.parsefile(joinpath(root, "paused.json"))
    @test paused["session"]["status"] == "paused"
    @test paused["execution"]["status"] == "running"
    @test paused["completed_updates"] == 3
    @test JSON.parsefile(joinpath(root, "completed.json"))["status"] == "completed"
    @test !JSON.parsefile(joinpath(root, "tree.json"))["truncated"]
end
