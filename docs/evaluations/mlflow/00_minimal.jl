# Start the local server described in README.md before running this file.
# Run these setup lines once before evaluating selections in VS Code.
import Pkg
Pkg.activate(@__DIR__; io=devnull)
Pkg.instantiate(; allow_autoprecomp=false)
using MLFlowClient

client = MLFlow("http://127.0.0.1:5097")
# A unique experiment name lets this example run repeatedly.
experiment = createexperiment(client, "Minimal example $(time_ns())")
trial = createrun(client, experiment; run_name="chi = 16",
    start_time=round(Int, 1000time()))

logparam(client, trial, "chi", "16")
for step in 1:3
    cost = 0.1^step # Stand-in for a cost calculated by your algorithm.
    # MLFlowClient 0.10.0 needs an explicit Unix-millisecond timestamp.
    logmetric(client, trial, "cost", cost; step, timestamp=round(Int, 1000time()))
end

updaterun(client, trial; status=RunStatus.FINISHED, end_time=round(Int, 1000time()))
@show trial.info.run_id
nothing
