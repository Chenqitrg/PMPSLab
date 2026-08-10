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

@testset "gaugefix! — mixed spectrum is split symmetrically" begin
    Random.seed!(20260810)
    mps = random_mps(ComplexF64, 4, 2, 3)
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100)
    L, R = PMPSLab._mpskit_gauges(mps, alg)

    # At every bond, C = L * R = U * S * V† is the mixed-canonical gauge.
    # Construct the Vidal center tensor Gamma explicitly, then distribute the
    # square root of each adjacent bond spectrum onto the site tensor:
    #
    #     B_i = sqrt(S_{i-1}) * Gamma_i * sqrt(S_i).
    #
    # This must reproduce the projector update performed by gaugefix!.
    mixed = map(eachindex(mps)) do bond
        pos = HalfInt(bond)
        U, S, Vh, _ = PMPSLab.svd_trunc(L[pos] * R[pos]; trunc=alg.trscheme)
        return (; U, S, Vh)
    end

    symmetric_tensors = map(eachindex(mps)) do i
        left_bond = HalfInt(i - 1)
        right_bond = HalfInt(i)
        left = mixed[mod1(i - 1, length(mps))]
        right = mixed[i]

        Gamma = PMPSLab._projector_mul_front(
            PMPSLab.pseudopow(left.S, -1) * left.U' * L[left_bond],
            mps[i]
        ) * R[right_bond] * right.Vh' * PMPSLab.pseudopow(right.S, -1)

        return PMPSLab._projector_front(
            sqrt(left.S) * PMPSLab._projector_tail(Gamma)
        ) * sqrt(right.S)
    end

    gauge_fixed = deepcopy(mps)
    gaugefix!(gauge_fixed, alg)
    for i in eachindex(mps)
        @test gauge_fixed[i] ≈ symmetric_tensors[i] atol=1e-10 rtol=1e-10
    end
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
