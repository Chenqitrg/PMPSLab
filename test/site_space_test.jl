using Test, PMPSLab, TensorKit, Random
import MPSKit

@testset "MPSKit site-space interface with nonuniform bonds" begin
    @test PMPSLab.left_virtualspace === MPSKit.left_virtualspace
    @test PMPSLab.right_virtualspace === MPSKit.right_virtualspace
    physical = [ℂ^2, ℂ^3, ℂ^2]
    right_spaces = [ℂ^2, ℂ^3, ℂ^4]
    for mps in (rand(Xoshiro(42), PeriodicMPS, ComplexF64, physical, right_spaces),
                one_mps(ComplexF64, physical, right_spaces))
        @test collect(MPSKit.right_virtualspace(mps)) == right_spaces
        @test collect(MPSKit.left_virtualspace(mps)) == circshift(right_spaces, 1)
        @test all(MPSKit.physicalspace(mps, i) == physical[i] for i in 1:3)
        @test MPSKit.right_virtualspace(mps, 0) == right_spaces[end]
        @test MPSKit.left_virtualspace(mps, 4) == right_spaces[end]
        @test_throws MethodError mps[3 // 2]
    end
end

@testset "Integer gauge sites include the closing bond and a one-site ring" begin
    for N in (1, 3), site in (0, N, N + 1)
        mps = rand(Xoshiro(42), PeriodicMPS, ComplexF64, N, 2, 2)
        before = dense(mps)
        V = MPSKit.right_virtualspace(mps, site)
        U = unitary(rand(Xoshiro(43), ComplexF64, V ← V))
        @test insert_gauge!(mps, site, U, U') === mps
        @test assertconsistency(mps) === nothing
        @test dense(mps) ≈ before
    end
end

@testset "No separate bond-label API" begin
    for name in (:HalfInt, :HalfPeriodicVector, :virtualspace, :virtualspaces)
        @test !isdefined(PMPSLab, name)
    end
end
