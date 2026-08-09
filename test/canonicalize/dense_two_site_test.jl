@testset "dense — TwoSiteMPS" begin
    @testset "N=1 (partial trace)" begin
        D, d = 2, 2
        mps = random_mps2(ComplexF64, 1, d, D)
        T = mps[1]
        result = dense(mps)
        @test length(codomain(result)) == 2
        @plansor manual[-1 -2; ()] := T[1 -1 -2; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "N=2 vs manual @plansor" begin
        D, d = 2, 2
        mps = random_mps2(ComplexF64, 2, d, D)
        result = dense(mps)
        # dense leg order: P_{1,a}, P_{1,b}, P_{2,a}, P_{2,b}
        @plansor manual[-1 -2 -3 -4; ()] := mps[1][circ -1 -2; mid] *
                                            mps[2][mid -3 -4; circ]
        @test norm(result - manual) < 1e-12
    end

    @testset "Z2Irrep N=2 vs manual @plansor" begin
        V = Z2Space(0 => 3, 1 => 2)
        P = Z2Space(0 => 2)
        mps = random_mps2(ComplexF64, 2, P, V)
        result = dense(mps)
        @plansor manual[-1 -2 -3 -4; ()] := mps[1][circ -1 -2; mid] *
                                            mps[2][mid -3 -4; circ]
        @test norm(result - manual) < 1e-12
    end
end
