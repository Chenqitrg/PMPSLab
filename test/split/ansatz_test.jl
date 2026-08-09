@testset "public split API" begin
    @test PMPSLab.split === Base.split
    @test all(name -> name in names(PMPSLab), (
        :PeriodicSplit, :RandomAnsatz, :OneAnsatz, :SVDAnsatz,
        :PeriodicGaugeAnsatz,
    ))
    @test SVDAnsatz().truncation isa TruncationStrategy
    @test PeriodicGaugeAnsatz().svd_pre isa SVDAnsatz
end

@testset "RandomAnsatz / OneAnsatz — construction and initial approximation" begin
    N = 2
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA = random_mps2(ComplexF64, N, P, V)

    for ansatz in (RandomAnsatz(V, 2N), OneAnsatz(V, 2N))
        @test length(ansatz.virtual_space) == 2N
        @test ansatz.virtual_space[HalfInt(1)] == V

        mpsB = initial_approximation(mpsA, ansatz)
        @test length(mpsB) == 2N
        @test assertconsistency(mpsB) === nothing
        @test scalartype(mpsB[1]) == ComplexF64
        for i in 1:2N
            @test left_virtualspace(mpsB[i]) == V
        end
        @test physicalspace(mpsB[1]) == P
        @test physicalspace(mpsB[2]) == P
    end

    @test_throws DimensionMismatch initial_approximation(mpsA, RandomAnsatz(V, 2N + 1))
end

@testset "split_site — SVD split of one 2-site tensor" begin
    D, d = 3, 2
    T = randn(ComplexF64, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D)
    trunc = TruncationStrategy()
    B1, B2 = split_site(T, trunc)

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

@testset "SVDAnsatz — initial approximation preserves periodic density" begin
    B0 = random_mps(ComplexF64, 4, 2, 2)
    A = pair_contraction(B0)
    mpsB = initial_approximation(A, SVDAnsatz(TruncationStrategy()))

    @test length(mpsB) == 4
    @test assertconsistency(mpsB) === nothing
    @test dense(mpsB) ≈ dense(A) rtol=1e-10
end
