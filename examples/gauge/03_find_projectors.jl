using PMPSLab: find_projectors
# Run from PMPSLab: julia --project=. examples/gauge/03_find_projectors.jl
# Compute a pair of matrices for every bond without changing the MPS.
using PMPSLab: ProjectorGaugeFix, find_projectors
using PMPSLab: PeriodicMPS
using MPSKit: right_virtualspace
using TensorKit: id, domain, codomain
using Test: @test

mps = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
before = copy.(collect(mps))
algorithm = ProjectorGaugeFix(; tol=1e-12, maxiter=100)
PR, PL = find_projectors(mps, algorithm)

@test all(mps[i] == before[i] for i in eachindex(mps))
for i in eachindex(mps)
    V = right_virtualspace(mps, i)
    # PR[i]: V ← W; PL[i]: W ← V, on the bond to the right of site i.
    @test codomain(PR[i])[1] == V
    @test domain(PL[i])[1] == V
    @test domain(PR[i]) == codomain(PL[i])
    @test PR[i] * PL[i] ≈ id(ComplexF64, V) atol=1e-9
end
# The returned containers have periodic indexing.
@test PR[5] === PR[1]
@show domain(PR[1]) codomain(PR[1])
nothing
