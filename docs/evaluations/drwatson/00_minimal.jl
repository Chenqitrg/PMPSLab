include(joinpath(@__DIR__, "setup.jl"))
using DrWatson
using JLD2

settings = (chi=16, tolerance=1e-10)
result = (cost=0.001,) # Replace this line with your own computation.

# A new directory means repeated computations never reuse an older result.
run_dir = mktempdir(workspace; prefix="run-", cleanup=false)
file = joinpath(run_dir, savename(settings, "jld2"))
tagsave(file, Dict("settings" => settings, "result" => result);
    gitpath=repository, warn=false)

saved = wload(file)
@show file saved["settings"] saved["result"] saved["gitcommit"]
nothing
