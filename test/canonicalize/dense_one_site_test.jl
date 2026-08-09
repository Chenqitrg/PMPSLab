@testset "dense — OneSiteMPS" begin
    @testset "N=1 (partial trace)" begin
        D, d = 2, 3
        mps = random_mps(ComplexF64, 1, d, D)
        T = mps[1]
        result = dense(mps)
        @test codomain(result)[1] == ℂ^d
        @plansor manual[-1; ()] := T[1 -1; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "N=2 vs manual @plansor" begin
        D, d = 2, 2
        mps = random_mps(ComplexF64, 2, d, D)
        result = dense(mps)
        @plansor manual[-1 -2; ()] := mps[1][1 -1; 2] * mps[2][2 -2; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "N=3 vs manual @plansor" begin
        D, d = 2, 2
        mps = random_mps(ComplexF64, 3, d, D)
        result = dense(mps)
        @plansor manual[-1 -2 -3; ()] := mps[1][1 -1; 2] * mps[2][2 -2; 3] *
                                        mps[3][3 -3; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "Float64 element type" begin
        D, d = 2, 2
        mps = random_mps(Float64, 2, d, D)
        result = dense(mps)
        @plansor manual[-1 -2; ()] := mps[1][1 -1; 2] * mps[2][2 -2; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "Z2Irrep N=3 vs manual @plansor" begin
        V = Z2Space(0 => 2, 1 => 2)
        P = Z2Space(0 => 2)
        mps = random_mps(ComplexF64, 3, P, V)
        result = dense(mps)
        @plansor manual[-1 -2 -3; ()] := mps[1][1 -1; 2] * mps[2][2 -2; 3] *
                                        mps[3][3 -3; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "D=1 trivial bond" begin
        d = 3
        mps = random_mps(ComplexF64, 3, d, 1)
        result = dense(mps)
        @test codomain(result) == (ℂ^d ⊗ ℂ^d ⊗ ℂ^d)
    end
end
