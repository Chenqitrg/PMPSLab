@testset "HalfInt" begin
    h = half(7)
    @test h == HalfInt(7//2)
    @test h + 2 == HalfInt(11//2)
    @test h - 1 == HalfInt(5//2)
end

@testset "MPSKit gauges — normalized on nonuniform virtual bonds" begin
    P = PeriodicVector(fill(ℂ^2, 4))
    V = HalfPeriodicVector(PeriodicVector([ℂ^2, ℂ^3, ℂ^4, ℂ^3]))
    mps = rand(PeriodicMPS, ComplexF64, P, V)
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100)

    L, R = PMPSLab._mpskit_gauges(mps, alg)

    @test L isa HalfPeriodicVector
    @test R isa HalfPeriodicVector
    for i in eachindex(mps)
        pos = half(2i + 1)
        bond = right_virtualspace(mps[i])
        @test codomain(L[pos])[1] == bond
        @test domain(L[pos])[1] == bond
        @test codomain(R[pos])[1] == bond
        @test domain(R[pos])[1] == bond
        @test norm(L[pos]) ≈ 1
        @test norm(R[pos]) ≈ 1
    end
end

@testset "MPSKit gauge adapter — QR/LQ iteration only" begin
    alg = ProjectorCanonical()
    @test !hasproperty(alg, :alg_eigsolve)
    @test !hasproperty(alg, :eig_miniter)
    @test PMPSLab._left_algorithm(alg).eig_miniter == typemax(Int)
    @test PMPSLab._right_algorithm(alg).eig_miniter == typemax(Int)
end

@testset "find_projectors — without input copies" begin
    mps = rand(PeriodicMPS, ComplexF64, 4, 2, 3)
    tensor_objects = [mps[i] for i in 1:length(mps)]
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100)
    PR, PL = find_projectors(mps, alg)

    # Input tensors are passed by reference without copying or mutation.
    @test all(mps[i] === tensor_objects[i] for i in 1:length(mps))
    for bond in 1:length(mps)
        pos = half(2bond + 1)
        @test PR[pos] * PL[pos] ≈ id(domain(PR[pos])) atol=1e-10
    end
end
