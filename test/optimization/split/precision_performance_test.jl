using PMPSLab: SplitEnvironmentPolicy, NuclearNorm
using Test, Random, Logging, TensorKit, PMPSLab
import ExperimentLab, LatticeIteratorLab, TensorLinearSolveLab, PeriodicProductsLab

@testset "NNR local solves preserve the tensor precision before storage" begin
    Random.seed!(714)
    for T in (Float32, ComplexF32, Float64, ComplexF64)
        target = rand(PeriodicMPS, T, 2, (2, 2), 1)
        algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=2),
            solver=TensorLinearSolveLab.CholeskySolve(),
            regularization=NuclearNorm(schedule=PMPSLab.GeometricDecay(initial=0.01, factor=0.5)))
        state = PMPSLab.initialize_split(target, algorithm)
        tensor, cost = PMPSLab._solve_local_update(state, first(state.pending_iteration))
        @test scalartype(tensor) === T
        @test isfinite(cost)
        result = with_logger(NullLogger()) do
            fetch(ExperimentLab.start_run(PMPSLab.split_steps(state); threaded=false))
        end
        @test all(scalartype(t) === T for t in result.mps)
        @test all(scalartype(t) === T for t in state.regularization.auxiliary)
        @test all(scalartype(t) === T for t in state.regularization.dual)
        @test result.progress.penalty == Float64(real(T)(0.005))
    end
end

@testset "NNR rejects penalties unrepresentable in the tensor precision" begin
    target = rand(PeriodicMPS, Float32, 2, (2, 2), 1)
    for initial in (1e-50, 1e100)
        algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=1),
            solver=TensorLinearSolveLab.CholeskySolve(),
            regularization=NuclearNorm(schedule=PMPSLab.GeometricDecay(; initial, minimum=0)))
        state = PMPSLab.initialize_split(target, algorithm)
        @test_throws ArgumentError PMPSLab._solve_local_update(state, first(state.pending_iteration))
    end
end

# Compare the complete production solve with the same Krylov solver acting
# directly on the unregularized transfer. Warm both before measuring allocations.
function unregularized_solve_allocations(solver, x, transfer, rhs)
    direct = () -> TensorLinearSolveLab.rdiv(solver, x,
        y -> PMPSLab._split_norm_action(y, transfer), rhs)
    wrapped = () -> PMPSLab._solve_split_norm(solver, x, transfer, rhs)
    direct(); wrapped()
    direct_bytes = minimum([@allocated direct() for _ in 1:3])
    wrapped_bytes = minimum([@allocated wrapped() for _ in 1:3])
    return direct_bytes, wrapped_bytes
end

@testset "Unregularized factored Krylov adds no tensor arithmetic" begin
    Random.seed!(715)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    policy = PMPSLab.TruncatedTransferPolicy(
        trunc=PMPSLab.MPSKit.TruncationStrategy())
    solver = TensorLinearSolveLab.KrylovSolve(
        algorithm=TensorLinearSolveLab.KrylovKit.GMRES(tol=1e-10, maxiter=100))
    algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=1), solver=solver,
        environments=SplitEnvironmentPolicy(normal=policy))
    state = PMPSLab.initialize_split(target, algorithm)
    transfer = PeriodicProductsLab.environment(state.normal_cache, 1)
    x = transpose(state.approximation[1], ((2,), (1, 3)))
    rhs = PMPSLab._split_norm_action(2x, transfer)
    direct_bytes, wrapped_bytes = unregularized_solve_allocations(solver, x, transfer, rhs)
    # Allow small wrapper overhead, but not extra tensor-sized temporaries.
    @test wrapped_bytes <= direct_bytes + 256
    @test PMPSLab._solve_split_norm(solver, x, transfer, rhs) ≈ 2x
end
