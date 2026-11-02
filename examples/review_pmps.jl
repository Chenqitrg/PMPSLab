# Optional runner for all independent PMPSLab examples.
# From PMPSLab: julia --project=. examples/review_pmps.jl
# For individual examples and reading order, see examples/README.md.

for file in (
    "01_construction.jl",
    "02_periodic_indexing.jl",
    "03_dense_contraction.jl",
    "04_pair_contraction.jl",
    "05_gauge_insertion.jl",
    "06_validation.jl",
    "07_tensor_mapping.jl",
    "08_finite_lift.jl",
    "09_symmetry.jl",
)
    println("\n── ", file, " ──")
    include(joinpath(@__DIR__, file))
end

println("\nAll PMPSLab examples passed.")
