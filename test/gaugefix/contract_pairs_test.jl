@testset "contract_pairs" begin
    @testset "structure — four inputs give two contracted tensors" begin
        D, d, N = 2, 2, 4
        mps = rand(PeriodicMPS, ComplexF64, N, d, D)
        paired = contract_pairs(mps)

        @test length(paired) == N ÷ 2
        for T in paired
            @test length(codomain(T)) == 3          # L_{2k-1} ⊗ P_{2k-1} ⊗ P_{2k}
            @test codomain(T)[1] == ℂ^D             # left virtual leg survives
            @test codomain(T)[2] == ℂ^d             # P_{2k-1}
            @test codomain(T)[3] == ℂ^d             # P_{2k}
            @test length(domain(T)) == 1
            @test domain(T)[1] == ℂ^D               # right virtual leg
        end
    end

    @testset "pair matches manual @plansor" begin
        D, d = 2, 2
        mps = rand(PeriodicMPS, ComplexF64, 2, d, D)
        paired = contract_pairs(mps)

        @test length(paired) == 1
        @plansor manual[-1 -2 -3; -4] := mps[1][-1 -2; 1] * mps[2][1 -3; -4]
        @test paired[1] ≈ manual
    end

    @testset "start site selects the periodic pairing" begin
        D, d, N = 2, 2, 4
        mps = rand(PeriodicMPS, ComplexF64, N, d, D)
        paired = contract_pairs(mps, 2)

        @plansor first_pair[-1 -2 -3; -4] :=
            mps[2][-1 -2; 1] * mps[3][1 -3; -4]
        @plansor wrapped_pair[-1 -2 -3; -4] :=
            mps[4][-1 -2; 1] * mps[1][1 -3; -4]

        @test paired[1] ≈ first_pair
        @test paired[2] ≈ wrapped_pair
        @test dense(paired) ≈ dense(circshift(mps, -1))
        @test all(
            contract_pairs(mps, N + 2)[site] ≈ paired[site]
            for site in eachindex(paired)
        )
    end

    @testset "dense invariant — N ∈ {2, 4, 6}" begin
        D, d = 2, 2
        for N in [2, 4, 6]
            mps = rand(PeriodicMPS, ComplexF64, N, d, D)
            # Same physical leg order as dense(mps); the periodic product is preserved.
            @test dense(contract_pairs(mps)) ≈ dense(mps)
        end
    end

    @testset "odd number of sites throws" begin
        D, d = 2, 2
        mps = rand(PeriodicMPS, ComplexF64, 3, d, D)
        @test_throws ArgumentError contract_pairs(mps)
    end

    @testset "Z2Irrep — dense invariant and structure" begin
        N = 4
        V = Z2Space(0 => 2, 1 => 2)
        P = Z2Space(0 => 2)
        mps = rand(PeriodicMPS, ComplexF64, N, P, V)
        paired = contract_pairs(mps)

        @test length(paired) == 2
        @test codomain(paired[1]) == V ⊗ P ⊗ P
        @test dense(paired) ≈ dense(mps)
    end

    @testset "Float64 element type" begin
        D, d, N = 2, 2, 4
        mps = rand(PeriodicMPS, Float64, N, d, D)
        paired = contract_pairs(mps)
        @test scalartype(paired[1]) == Float64
        @test dense(paired) ≈ dense(mps)
    end
end
