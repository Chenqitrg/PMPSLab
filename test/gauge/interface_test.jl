@testset "MPSKit gauges — normalized on nonuniform virtual bonds" begin
    P = PeriodicVector(fill(ℂ^2, 4))
    V = PeriodicVector([ℂ^2, ℂ^3, ℂ^4, ℂ^3])
    mps = rand(PeriodicMPS, ComplexF64, P, V)
    algorithm = ProjectorGaugeFix(; tol=1e-12, maxiter=100)

    L, R = PMPSLab._mpskit_gauges(mps, algorithm)

    @test L isa PeriodicVector
    @test R isa PeriodicVector
    for i in eachindex(mps)
        pos = i
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
    algorithm = ProjectorGaugeFix()
    @test !hasproperty(algorithm, :alg_eigsolve)
    @test !hasproperty(algorithm, :eig_miniter)
    @test PMPSLab._left_algorithm(algorithm).eig_miniter == typemax(Int)
    @test PMPSLab._right_algorithm(algorithm).eig_miniter == typemax(Int)
end

@testset "find_projectors — without input copies" begin
    mps = rand(PeriodicMPS, ComplexF64, 4, 2, 3)
    tensor_objects = [mps[i] for i in 1:length(mps)]
    algorithm = ProjectorGaugeFix(; tol=1e-12, maxiter=100)
    PR, PL = find_projectors(mps, algorithm)

    # Input tensors are passed by reference without copying or mutation.
    @test all(mps[i] === tensor_objects[i] for i in 1:length(mps))
    for bond in 1:length(mps)
        pos = bond
        @test PR[pos] * PL[pos] ≈ id(domain(PR[pos])) atol=1e-10
    end
end
