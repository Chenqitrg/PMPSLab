using PMPSLab: SVDInitialization
using PMPSLab
import ExperimentLab, LatticeIteratorLab, TensorLinearSolveLab
using TensorKit: truncrank
using Random: seed!

seed!(1234)
target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
algorithm = ALSSplit(initialization=SVDInitialization(trunc=truncrank(1)),
    sweep=LatticeIteratorLab.CircularSweep(max_sweeps=1), solver=TensorLinearSolveLab.LUSolve())
state = PMPSLab.initialize_split(target, algorithm)

# the split algorithm performs one complete site update per yield. ExperimentLab decides
# whether the generator may continue to the next site.
stop_after_site = 2
session = ExperimentLab.start_run(ExperimentLab.algorithm_steps(state); threaded=false,
    onstep=(session, entry) -> begin
        @assert entry === state.progress # The callback sees the same progress value.
        println("Completed site ", entry.site, ", cost = ", entry.local_cost)
        if entry.sweep == 1 && entry.site == stop_after_site
            ExperimentLab.pause!(session)
        end
    end)

# Wait for the pause to be acknowledged. Site 2 and its cache refresh are done;
# site 3 has not started. The Julia process stays alive and tensors stay in memory.
@assert timedwait(() -> ExperimentLab.run_status(session) == ExperimentLab.PAUSED, 20) == :ok
@assert state.progress.completed_updates == 2
@show ExperimentLab.run_status(session) # PAUSED
@show state.progress

# This file intentionally leaves the session paused for REPL inspection.
# Execute these lines yourself to continue from site 3:
# ExperimentLab.resume!(session)
# result = fetch(session)
# result.progress.completed_updates  # 4
#
# fetch(session) alone waits for completion; it does not resume a paused run.
# Alternatively, abandon the remaining work with ExperimentLab.cancel!(session).
nothing
