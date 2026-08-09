@testset "random_mps / one_mps" begin
    mps = random_mps(ComplexF64, 4, 2, 3)
    @test length(mps) == 4
    @test assertconsistency(mps) === nothing
    @test scalartype(mps[1]) == ComplexF64
    @test codomain(mps[1])[1] == ℂ^3           # virtual ℂ^D
    @test codomain(mps[1])[2] == ℂ^2           # physical ℂ^d
    @test domain(mps[1])[1] == ℂ^3

    # deterministic: one_mps is the uniform all-ones MPS
    mps_a = one_mps(ComplexF64, 3, 2, 2)
    mps_b = one_mps(ComplexF64, 3, 2, 2)
    @test all(mps_a[i] ≈ mps_b[i] for i in 1:3)
    @test assertconsistency(one_mps(Float64, 4, 2, 2)) === nothing
    @test scalartype(one_mps(Float64, 4, 2, 2)[1]) == Float64

    # graded spaces
    V = Z2Space(0 => 2, 1 => 2)
    P = Z2Space(0 => 2)
    mps_g = random_mps(ComplexF64, 3, P, V)
    @test length(mps_g) == 3
    @test assertconsistency(mps_g) === nothing
    @test assertconsistency(one_mps(ComplexF64, 3, P, V)) === nothing
end

@testset "random_mps2 / one_mps2 — two-site tensors" begin
    mps = random_mps2(ComplexF64, 4, 2, 3)
    @test length(mps) == 4
    @test assertconsistency(mps) === nothing
    @test scalartype(mps[1]) == ComplexF64
    @test numout(mps[1]) == 3
    @test codomain(mps[1])[1] == ℂ^3           # virtual ℂ^D
    @test codomain(mps[1])[2] == ℂ^2           # physical ℂ^d (site a)
    @test codomain(mps[1])[3] == ℂ^2           # physical ℂ^d (site b)
    @test domain(mps[1])[1] == ℂ^3

    # deterministic: one_mps2 is the uniform all-ones two-site MPS
    mps_a = one_mps2(ComplexF64, 3, 2, 2)
    mps_b = one_mps2(ComplexF64, 3, 2, 2)
    @test all(mps_a[i] ≈ mps_b[i] for i in 1:3)
    @test assertconsistency(one_mps2(Float64, 4, 2, 2)) === nothing
    @test scalartype(one_mps2(Float64, 4, 2, 2)[1]) == Float64

    # graded spaces
    V = Z2Space(0 => 2, 1 => 2)
    P = Z2Space(0 => 2)
    mps_g = random_mps2(ComplexF64, 3, P, V)
    @test length(mps_g) == 3
    @test assertconsistency(mps_g) === nothing
    @test assertconsistency(one_mps2(ComplexF64, 3, P, V)) === nothing
end

@testset "assertconsistency — valid and invalid" begin
    D = 3
    mps = random_mps(ComplexF64, 4, 2, D)
    tvec = [copy(mps[i]) for i in 1:4]
    @test assertconsistency(mps) === nothing

    # Mismatched: change one site's right virtual space
    tvec_bad = copy(tvec)
    tvec_bad[2] = TensorMap(randn(ComplexF64, D*2, D+1), ℂ^D ⊗ ℂ^2, ℂ^(D+1))
    mps_bad = PeriodicMPS(PeriodicVector(tvec_bad))
    @test_throws TensorKit.SpaceMismatch assertconsistency(mps_bad)
end

@testset "assertconsistency — Z2Irrep" begin
    V = Z2Space(0 => 3, 1 => 2)
    P = Z2Space(0 => 2)
    mps = random_mps(ComplexF64, 4, P, V)
    tvec = [copy(mps[i]) for i in 1:4]
    @test assertconsistency(mps) === nothing
    tvec_bad = copy(tvec)
    tvec_bad[2] = rand(ComplexF64, V ⊗ P ← Z2Space(0 => 3, 1 => 3))
    mps_bad = PeriodicMPS(PeriodicVector(tvec_bad))
    @test_throws TensorKit.SpaceMismatch assertconsistency(mps_bad)
end
