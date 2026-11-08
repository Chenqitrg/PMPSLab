using PMPSLab: RandomInitialization, IdentityInitialization, SVDInitialization, GaugeInitialization, SplitEnvironmentPolicy
@testset "public split API" begin
    @test PMPSLab.split === Base.split
    @test all(name -> name in names(PMPSLab), (
        :ALSSplit, :RandomInitialization, :IdentityInitialization, :SVDInitialization,
        :GaugeInitialization, :SplitEnvironmentPolicy,
    ))
    @test all(name -> Base.ispublic(PMPSLab, name), (
        :AbstractTransferPolicy, :DenseTransferPolicy, :TruncatedTransferPolicy,
    ))
    @test SVDInitialization().trunc isa TruncationStrategy
    @test GaugeInitialization().initialization isa SVDInitialization
end

@testset "RandomInitialization / IdentityInitialization — construction and initial approximation" begin
    N = 2
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA = rand(PeriodicMPS, ComplexF64, N, P ⊗ P, V)

    for initialization in (
            RandomInitialization(; virtual_spaces=fill(V, 2N)),
            IdentityInitialization(; virtual_spaces=fill(V, 2N)),
        )
        @test length(initialization.virtual_spaces) == 2N
        @test initialization.virtual_spaces[1] == V

        mpsB = initial_approximation(mpsA, initialization)
        @test length(mpsB) == 2N
        @test assertconsistency(mpsB) === nothing
        @test scalartype(mpsB[1]) == ComplexF64
        for i in 1:2N
            @test left_virtualspace(mpsB[i]) == V
        end
        @test physicalspace(mpsB[1]) == P
        @test physicalspace(mpsB[2]) == P
    end

    @test_throws DimensionMismatch initial_approximation(
        mpsA,
        RandomInitialization(; virtual_spaces=fill(V, 2N + 1)),
    )
end

@testset "split_tensor — SVD split of one two-physical-leg tensor" begin
    D, d = 3, 2
    T = randn(ComplexF64, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D)
    trunc = TruncationStrategy()
    B1, B2 = split_tensor(T, trunc)

    @test length(codomain(B1)) == 2
    @test codomain(B1)[1] == ℂ^D
    @test codomain(B1)[2] == ℂ^d
    @test length(domain(B1)) == 1
    @test length(codomain(B2)) == 2
    @test codomain(B2)[1] == domain(B1)[1]
    @test codomain(B2)[2] == ℂ^d
    @test domain(B2)[1] == ℂ^D

    @plansor recon[-1 -2 -3; -4] := B1[-1 -2; 1] * B2[1 -3; -4]
    @test recon ≈ T rtol=1e-12
end

@testset "SVDInitialization — initial approximation preserves periodic density" begin
    B0 = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    A = contract_pairs(B0)
    mpsB = initial_approximation(A, SVDInitialization(; trunc=TruncationStrategy()))

    @test length(mpsB) == 4
    @test assertconsistency(mpsB) === nothing
    @test dense(mpsB) ≈ dense(A) rtol=1e-10
end

@testset "Ansatz virtual spaces follow MPSKit right-site convention" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    spaces = [ℂ^2, ℂ^3, ℂ^4, ℂ^3]
    for initialization in (RandomInitialization(spaces), IdentityInitialization(; virtual_spaces=spaces))
        approximation = initial_approximation(target, initialization)
        @test collect(right_virtualspace(approximation)) == spaces
        @test collect(left_virtualspace(approximation)) == circshift(spaces, 1)
        @test initialization.virtual_spaces isa PeriodicVector
    end
end
