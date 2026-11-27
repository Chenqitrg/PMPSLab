# Run in an experiment environment with PMPSLab, DrWatson, JLD2 and JSON.
# CDL_JOURNEY_OUTPUT may name an existing experiment output directory.
using PMPSLab, DrWatson, JLD2, JSON, Random, LinearAlgebra, Test
import AlgorithmsInterface as AI
import LatticeIteratorLab, TensorLinearSolveLab
include(joinpath(@__DIR__, "journeys", "cdl_diagnostics.jl"))

output = get(ENV, "CDL_JOURNEY_OUTPUT", mktempdir())
fixture = CDLFixture()
Random.seed!(fixture.seed)
target = random_cdl_mps(ComplexF64, fixture.sites, fixture.corner_dimension)
if fixture.normalize
    scale = norm(dense(target))^(1 / fixture.sites)
    target = PeriodicMPS([tensor / scale for tensor in target])
end
initial = initial_approximation(target, PMPSLab.SVDInitialization())
before = measure(CDLDiagnostics(), (; target, mps=initial))
@test before.max_second_ratio > 0.1
@test before.relative_squared_error < 1e-20
wsave(joinpath(output, "input.jld2"), Dict("target" => target, "initial" => initial, "fixture" => fixture))

solver = TensorLinearSolveLab.KrylovSolve(
    algorithm=TensorLinearSolveLab.KrylovKit.GMRES(tol=1e-12, maxiter=100))
cases = (
    "als" => PMPSLab.NoRegularization(),
    "nnr_default" => PMPSLab.NuclearNorm(),
    "nnr_slow_decay" => PMPSLab.NuclearNorm(
        schedule=PMPSLab.GeometricDecay(initial=0.01, factor=0.95)),
)
problem = PMPSLab.SplitProblem(target)
for (label, regularization) in cases
    algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=100), solver=solver,
        regularization=regularization)
    state = AI.initialize_state(problem, algorithm; iterate=initial)
    events = PMPSLab.SplitProgress[]
    measurements = [(step=0, data=before)]
    while !AI.is_finished!(problem, algorithm, state)
        AI.increment!(state)
        AI.step!(problem, algorithm, state)
        progress = state.progress
        push!(events, progress)
        if progress.sweep_completed && progress.completed_sweeps in (1, 5, 10, 20, 50, 100)
            push!(measurements, (step=progress.completed_updates,
                data=measure(CDLDiagnostics(), (; target, mps=state.iterate))))
            if label == "nnr_slow_decay" && progress.completed_sweeps == 50
                wsave(joinpath(output, "checkpoint.jld2"), Dict("snapshot" => PMPSLab.snapshot(state)))
            end
        end
    end
    result = AI.finalize_state!(problem, algorithm, state)
    final = last(measurements).data
    if label == "nnr_slow_decay"
        @test maximum(final.effective_pair_ranks) == 1
        @test final.relative_squared_error < 1e-12
        @test final.rank1_relative_change < 1e-16
    end
    # The application supplies input/run links; DrWatson stores objects and Git provenance.
    tagsave(joinpath(output, "$label.jld2"), Dict("algorithm" => algorithm,
        "input" => "input.jld2", "mps" => result.mps, "measurements" => measurements);
        gitpath=normpath(joinpath(@__DIR__, "..", "..")), storepatch=true)
    write(joinpath(output, "$label.json"), JSON.json((; label, parent_run=nothing,
        checkpoint_step=0, events, measurement=final)))
end

saved = wload(joinpath(output, "checkpoint.jld2"))["snapshot"]
recovered = AI.solve(problem, saved.algorithm; snapshot=saved)
expected = wload(joinpath(output, "nnr_slow_decay.jld2"))["mps"]
restart_difference = norm(dense(recovered.mps) - dense(expected))^2 / norm(dense(expected))^2
@test restart_difference < 1e-18
write(joinpath(output, "resumed.json"), JSON.json((; label="resumed", parent_run="nnr_slow_decay",
    checkpoint_step=saved.progress.completed_updates, relative_squared_difference=restart_difference,
    measurement=measure(CDLDiagnostics(), (; target, mps=recovered.mps)))))
println("JSON reader: python3 examples/optimization/journeys/read_cdl_json.py ", repr(output))
@show output
nothing
