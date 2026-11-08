if "journey" in ARGS
    include("journeys/runtests.jl")
else
    include("numerical.jl")
    if !("numerical" in ARGS)
        include("ambiguity_test.jl")
        include("package_boundary_test.jl")
        include("execution/runtests.jl")
        include("journeys/runtests.jl")
    end
end
