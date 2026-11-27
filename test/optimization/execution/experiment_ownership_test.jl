using Test, PMPSLab, TensorKit, LatticeIteratorLab, TensorLinearSolveLab

@testset "A.I. callbacks observe committed site updates" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=ZigzagSweep(max_sweeps=1), solver=CholeskySolve())
    events = PMPSLab.SplitProgress[]
    snapshots = PMPSLab.SplitSnapshot[]
    observe = function observe(::PMPSLab.SplitProblem, ::ALSSplit, state::PMPSLab.SplitState)::Nothing
        push!(events, state.progress)
        state.progress.completed_updates == 3 && push!(snapshots, PMPSLab.snapshot(state))
        return nothing
    end
    result = AI.with_algorithmlogger(
        function run_observed()::NamedTuple{(:mps,:progress)}
            return AI.solve(PMPSLab.SplitProblem(target), algorithm)
        end,
        :PostStep => AI.CallbackAction(observe),
    )
    @test getfield.(events, :site) == [1, 2, 3, 4, 3, 2]
    @test events[end].sweep_completed
    @test all(isfinite(event.local_cost) for event in events)
    saved = only(snapshots)
    recovered = AI.solve(PMPSLab.SplitProblem(target), algorithm; snapshot=saved)
    @test dense(recovered.mps) ≈ dense(result.mps)
    @test recovered.progress.completed_updates == 6
    @test saved.progress.completed_updates == 3
end
