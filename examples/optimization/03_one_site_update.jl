# Run from PMPSLab: julia --project=. examples/optimization/03_one_site_update.jl
using PMPSLab: SVDInitialization
using PMPSLab
import LatticeIteratorLab, TensorLinearSolveLab
import PeriodicProductsLab
import AlgorithmsInterface as AI
using TensorKit: truncrank
using Random: seed!
using Test: @test

seed!(1234)
target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
algorithm = ALSSplit(
    initialization=SVDInitialization(trunc=truncrank(1)),
    sweep=LatticeIteratorLab.CircularSweep(max_sweeps=1),
    solver=TensorLinearSolveLab.LUSolve(),
)
state = PMPSLab.initialize_split(target, algorithm)
before = deepcopy(state.iterate)

# Inspect the workspace for this review. Application code can simply call split.
# PMPSLab constructs site transfers from MPS tensors.
transfer = PMPSLab.transfer_at(before, 1)
@test transfer isa PMPSLab.DenseTransferMatrix

# initialize_split has already built two PeriodicProductsLab caches through
# PMPSLab.build_transfer_cache: normal <B|B> and mixed <B|A>.
normal_cache = state.normal_cache
mixed_cache = state.mixed_cache
@test normal_cache isa PeriodicProductsLab.PeriodicCache
@test mixed_cache isa PeriodicProductsLab.PeriodicCache
normal_environment = PeriodicProductsLab.environment(normal_cache, 1)
@test normal_environment isa PMPSLab.DenseTransferMatrix
@show transfer normal_cache

# Perform exactly one site update through the public A.I. protocol.
problem = PMPSLab.SplitProblem(target)
initial_error = PMPSLab.squared_distance(target, before)
@test !AI.is_finished!(problem, algorithm, state)
AI.increment!(state)
AI.step!(problem, algorithm, state)
entry = state.progress
exact_error = PMPSLab.squared_distance(target, state.iterate)
@test entry.site == 1
@test entry.completed_updates == 1
@test all(state.iterate[i] == before[i] for i in 2:4)
@test exact_error <= initial_error + 1e-9 * max(initial_error, 1.0)
@test isapprox(entry.local_cost, exact_error; atol=1e-9, rtol=1e-9)
@show entry.site initial_error exact_error
# No background task advances the state after this script returns.
nothing
