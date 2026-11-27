using PMPSLab: SVDInitialization, NuclearNorm
using Test, Random, LinearAlgebra, PMPSLab, JLD2, JSON
import AlgorithmsInterface as AI
import LatticeIteratorLab, TensorLinearSolveLab
include(joinpath(@__DIR__, "..", "..", "..", "examples", "optimization", "journeys", "cdl_diagnostics.jl"))
include("json_contract.jl")

# The application records progress and chooses when to persist a numerical checkpoint.
function cdl_run!(state::PMPSLab.SplitState;
        checkpoint_file::Union{Nothing,AbstractString}=nothing, checkpoint_step::Int=0,
    )::Vector{PMPSLab.SplitProgress}
    events = PMPSLab.SplitProgress[]
    problem, algorithm = PMPSLab.SplitProblem(state.target), state.algorithm
    while !AI.is_finished!(problem, algorithm, state)
        AI.increment!(state)
        AI.step!(problem, algorithm, state)
        push!(events, state.progress)
        if !isnothing(checkpoint_file) && state.progress.completed_updates == checkpoint_step
            jldsave(checkpoint_file; snapshot=PMPSLab.snapshot(state))
        end
    end
    return events
end

@testset "CDL NNR journey: native execution, restart and JSON observation" begin
    Random.seed!(716)
    target = random_cdl_mps(ComplexF64, 4, 2)
    scale = norm(dense(target))^(1 / length(target))
    target = PeriodicMPS([tensor / scale for tensor in target])
    original_target = deepcopy(target)
    initial = initial_approximation(target, SVDInitialization())
    original_initial = deepcopy(initial)
    before = measure(CDLDiagnostics(), (; target, mps=initial))
    @test before.relative_squared_error < 1e-20
    @test before.max_second_ratio > 0.1
    @test before.rank1_relative_change > 1e-4
    sweeps, sites = 50, length(initial)
    steps = sweeps * sites
    solver = TensorLinearSolveLab.KrylovSolve(
        algorithm=TensorLinearSolveLab.KrylovKit.GMRES(tol=1e-12, maxiter=100))

    mktempdir(function check_journey(path::String)::Nothing
        for (label, regularization) in (
                "als" => PMPSLab.NoRegularization(),
                "nnr" => NuclearNorm(schedule=PMPSLab.GeometricDecay(initial=0.01, factor=0.95)),
            )
            algorithm = ALSSplit(; sweep=LatticeIteratorLab.CircularSweep(max_sweeps=sweeps), solver, regularization)
            state = AI.initialize_state(PMPSLab.SplitProblem(target), algorithm; iterate=initial)
            file = joinpath(path, "checkpoint.jld2")
            events = cdl_run!(state; checkpoint_file=label == "nnr" ? file : nothing,
                checkpoint_step=steps ÷ 2)
            after = measure(CDLDiagnostics(), (; target, mps=state.iterate))
            @test after.bond_dimensions == before.bond_dimensions
            @test after.relative_squared_error < 1e-10
            if label == "nnr"
                @test all(==(1), after.effective_pair_ranks)
                @test after.max_second_ratio < 1e-6
                @test after.rank1_relative_change < 1e-16
                restored = PMPSLab.restore_split(load(file, "snapshot"), target)
                @test restored.progress.completed_updates == steps ÷ 2
                resumed_events = cdl_run!(restored)
                @test norm(dense(restored.iterate) - dense(state.iterate))^2 / norm(dense(state.iterate))^2 < 1e-18
                resumed = measure(CDLDiagnostics(), (; target, mps=restored.iterate))
                write(joinpath(path, "resumed.json"), JSON.json((;
                    label="resumed", parent_run="nnr", checkpoint_step=steps ÷ 2,
                    events=resumed_events, measurement=resumed)))
            end
            write(joinpath(path, "$label.json"), JSON.json((;
                label, parent_run=nothing, checkpoint_step=0, events, measurement=after)))
        end
        check_cdl_json(path; sweeps, sites)
        return nothing
    end)
    @test all(target[i] == original_target[i] for i in eachindex(target))
    @test all(initial[i] == original_initial[i] for i in eachindex(initial))
end
