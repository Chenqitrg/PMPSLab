@testset "gaugefix! — integration" begin
    N, d, D = 4, 2, 3
    mps = random_mps(ComplexF64, N, d, D)
    tvec = [copy(mps[i]) for i in 1:N]
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())
    @test gaugefix!(mps, alg) === mps
    # After canonicalization, consistency should still hold
    @test assertconsistency(mps) === nothing
    # Tensors should have been modified
    for i in 1:N
        @test !(mps[i] ≈ tvec[i])
    end
end

@testset "gaugefix! — Z2Irrep" begin
    N = 4
    V = Z2Space(0 => 3, 1 => 2)
    P = Z2Space(0 => 2)
    mps = random_mps(ComplexF64, N, P, V)
    tvec = [copy(mps[i]) for i in 1:N]
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())
    @test gaugefix!(mps, alg) === mps
    @test assertconsistency(mps) === nothing
    for i in 1:N
        @test !(mps[i] ≈ tvec[i])
    end
end

@testset "gaugefix! — self-consistency" begin
    N, d, D = 4, 2, 2
    mps = random_mps(ComplexF64, N, d, D)
    dense_before = dense(mps)
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())
    gaugefix!(mps, alg)
    dense_after = dense(mps)
    @test norm(dense_before - dense_after) < 1e-10
end

function tensor_relative_distance(mps1::PeriodicMPS, mps2::PeriodicMPS)
    numerator = sum(norm(mps1[i] - mps2[i])^2 for i in eachindex(mps1))
    denominator = sum(norm(mps1[i])^2 for i in eachindex(mps1))
    return sqrt(numerator / denominator)
end

function state_relative_distance(mps1::PeriodicMPS, mps2::PeriodicMPS)
    state1 = dense(mps1)
    state2 = dense(mps2)
    return norm(state1 - state2) / norm(state1)
end

@testset "gaugefix! — numerically idempotent without truncation" begin
    Random.seed!(20260807)
    V = Z2Space(0 => 3, 1 => 2)
    P = Z2Space(0 => 2)
    cases = (
        random_mps(ComplexF64, 4, 2, 3),
        random_mps2(ComplexF64, 4, 2, 3),
        random_mps(ComplexF64, 4, P, V),
    )
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100)

    for original in cases
        once = deepcopy(original)
        gaugefix!(once, alg)
        twice = deepcopy(once)
        gaugefix!(twice, alg)

        @test tensor_relative_distance(once, twice) < 1e-10
        @test state_relative_distance(once, twice) < 1e-12
    end
end

@testset "gaugefix! — truncation is physically idempotent" begin
    Random.seed!(20260808)
    original = random_mps(ComplexF64, 6, 2, 6)
    alg = ProjectorCanonical(
        ; tol=1e-12, maxiter=100,
        trscheme=MPSKit.TruncationStrategy(maxrank=2),
    )

    once = deepcopy(original)
    gaugefix!(once, alg)
    twice = deepcopy(once)
    gaugefix!(twice, alg)
    thrice = deepcopy(twice)
    gaugefix!(thrice, alg)

    # The first pass may both compress and change gauge. Do not require its raw
    # tensors to equal the second pass: only the represented state and retained
    # bond spaces must already be stable. Once compressed, the next result is a
    # tensor-level fixed point up to the solver tolerance.
    @test all(dim(left_virtualspace(once[i])) == 2 for i in eachindex(once))
    @test all(space(twice[i]) == space(once[i]) for i in eachindex(once))
    @test all(space(thrice[i]) == space(twice[i]) for i in eachindex(twice))
    @test state_relative_distance(once, twice) < 1e-12
    @test state_relative_distance(twice, thrice) < 1e-12
    @test tensor_relative_distance(twice, thrice) < 1e-10
end
