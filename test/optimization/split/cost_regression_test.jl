using LinearAlgebra: norm
using Test, Random, PMPSLab, TensorKit

# Deliberately stop the solve at the initial tensor. The objective must remain
# correct even when the normal equation has not been solved.
struct IncompleteSolve <: TensorLinearSolveLab.AbstractSolve end
TensorLinearSolveLab.rdiv(::IncompleteSolve, initial, normal, rhs) = initial

@testset "local objective remains valid for an incomplete solve" begin
    Random.seed!(415)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    approximation = rand(PeriodicMPS, ComplexF64, 4, 2, 1)
    ctx = test_split_state(target, approximation; solver=IncompleteSolve())
    state = LatticeIteratorLab.SweepState{LatticeIteratorLab.Forward}(1, 0)
    tensor, cost = PMPSLab._solve_local_update(ctx, state)
    candidate = deepcopy(approximation)
    candidate[1] = tensor
    @test cost ≈ norm(dense(target) - dense(candidate))^2
end
