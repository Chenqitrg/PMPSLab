# Run this once before evaluating selected lines in VS Code.
import Pkg
Pkg.activate(@__DIR__; io=devnull)
Pkg.instantiate(; allow_autoprecomp=false)

using MLFlowClient
using JLD2
using JSON
using Logging
using LoggingExtras
import LoggingFormats
using TensorKit
using LinearAlgebra
using UUIDs

workspace = abspath(get(ENV, "MLFLOW_REVIEW_WORKSPACE", joinpath(@__DIR__, "workspace")))
mkpath(workspace)
client = MLFlow(get(ENV, "MLFLOW_REVIEW_URL", "http://127.0.0.1:5097"))
