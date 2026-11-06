# Run once before evaluating selected lines in the REPL.
import Pkg
Pkg.activate(@__DIR__; io=devnull)
Pkg.instantiate(; allow_autoprecomp=false)
workspace = joinpath(@__DIR__, "workspace")
repository = normpath(joinpath(@__DIR__, "..", "..", ".."))
mkpath(workspace)
