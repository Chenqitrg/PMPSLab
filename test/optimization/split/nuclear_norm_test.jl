using PMPSLab: NuclearNorm
using Test, LinearAlgebra, Random, TensorKit, PMPSLab
import LatticeIteratorLab, TensorLinearSolveLab, ExperimentLab

@testset "Nuclear norm configuration is available" begin
    @test isdefined(PMPSLab, :NuclearNorm)
    @test isdefined(PMPSLab, :GeometricDecay)
end

@testset "Nuclear norm proximal map soft-thresholds singular values" begin
    tensor = TensorMap(Matrix(Diagonal([3.0, 1.0, 0.2])), ℂ^3 ← ℂ^3)
    original = copy(tensor)
    actual = PMPSLab._soft_threshold(tensor, 0.5)
    expected = TensorMap(Matrix(Diagonal([2.5, 0.5, 0.0])), ℂ^3 ← ℂ^3)
    @test actual ≈ expected
    @test space(actual) == space(tensor)
    @test tensor == original
    @test norm(PMPSLab._soft_threshold(tensor, 4.0)) == 0
end

@testset "NNR normalization matches explicitly scaled tensors" begin
    Random.seed!(710)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    target[1] *= 11
    initial = rand(PeriodicMPS, ComplexF64, 4, 2, 1)
    before = deepcopy(target)
    C = real(PMPSLab.expectation(target, target))
    s = C^(1 / (2length(initial)))
    normalized_target = PeriodicMPS([t / s^2 for t in target])
    normalized_initial = PeriodicMPS([t / s for t in initial])
    algorithm = ALSSplit(
        sweep=LatticeIteratorLab.CircularSweep(max_sweeps=2),
        solver=TensorLinearSolveLab.CholeskySolve(),
        regularization=NuclearNorm(schedule=PMPSLab.GeometricDecay(initial=0.01, factor=0.8, minimum=1e-7)),
        error_measure=:exact,
    )
    actual = split(target, initial, algorithm)
    expected = split(normalized_target, normalized_initial, algorithm)
    @test dense(actual.mps) ≈ sqrt(C) * dense(expected.mps)
    @test all(target[i] == before[i] for i in eachindex(target))
    @test actual.progress.local_cost ≈ expected.progress.local_cost atol=1e-10
    @test actual.progress.exact_error ≈ norm(dense(target) - dense(actual.mps))^2 / C atol=1e-10
    @test actual.progress.cost_scale ≈ C
    @test actual.progress.penalty ≈ 0.008
    inplace = deepcopy(initial)
    @test split(target, inplace, algorithm; copy_initial=false).mps === inplace
    @test dense(inplace) ≈ dense(actual.mps)
end

@testset "NNR decay validates inputs and clamps at the configured floor" begin
    for keywords in ((; initial=0), (; factor=0), (; factor=1.1), (; minimum=-1),
                     (; initial=1e-8), (; factor=NaN), (; initial=Inf))
        @test_throws ArgumentError PMPSLab.GeometricDecay(; keywords...)
    end
    rule = NuclearNorm(schedule=PMPSLab.GeometricDecay(initial=0.1, factor=0.5, minimum=0.02))
    @test [PMPSLab._penalty(rule, k) for k in 0:3] ≈ [0.1, 0.05, 0.025, 0.02]
    zero_target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    zero_target[1] = zero(zero_target[1])
    algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=1),
        solver=TensorLinearSolveLab.CholeskySolve(), regularization=rule)
    initial = rand(PeriodicMPS, ComplexF64, 4, 2, 1)
    @test_throws ArgumentError PMPSLab.initialize_split(zero_target, initial, algorithm)
end

@testset "NNR thresholds the inserted pair bond and preserves graded spaces" begin
    Random.seed!(711)
    tensor = randn(ComplexF64, ℂ^2 ⊗ ℂ^3 ← ℂ^4)
    threshold = 0.7
    for site in 1:2
        result = PMPSLab._threshold_pair_bond(tensor, site, threshold)
        rows = isodd(site) ? 6 : 2
        expected = max.(svdvals(reshape(convert(Array, tensor), rows, :)) .- threshold, 0)
        @test svdvals(reshape(convert(Array, result), rows, :)) ≈ expected atol=1e-12
        @test space(result) == space(tensor)
    end
    V = Z2Space(0=>2, 1=>1)
    P = Z2Space(0=>1, 1=>1)
    graded = randn(ComplexF64, V ⊗ P ← V)
    for site in 1:2
        result = PMPSLab._threshold_pair_bond(graded, site, 0.1)
        @test space(result) == space(graded)
        @test norm(result) < norm(graded)
        @test norm(PMPSLab._threshold_pair_bond(graded, site, 2norm(graded))) == 0
    end
end
