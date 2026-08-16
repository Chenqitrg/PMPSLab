@testset "dot — agrees with expectation of PeriodicTransferMatrix" begin
    N, d, D = 3, 2, 2
    bra = rand(PeriodicMPS, ComplexF64, N, d, D)
    ket = rand(PeriodicMPS, ComplexF64, N, d, D)

    @test dot(bra, ket) ≈ expectation(PeriodicTransferMatrix(bra, ket))
    @test dot(bra, ket) ≈ overlap(bra, ket)
    @test dot(bra, bra) ≈ expectation(PeriodicTransferMatrix(bra))
    @test imag(dot(bra, bra)) ≈ 0 atol=1e-12
    @test real(dot(bra, bra)) > 0
end

@testset "norm — real, non-negative, analytic for one_mps" begin
    N, d, D = 4, 2, 2
    mps = rand(PeriodicMPS, ComplexF64, N, d, D)
    @test norm(mps)^2 ≈ dot(mps, mps)
    @test squared_norm(mps) ≈ real(dot(mps, mps))
    @test norm(mps) > 0
    @test imag(norm(mps)) ≈ 0 atol=1e-12

    mps1 = one_mps(ComplexF64, N, d, D)
    @test norm(mps1) ≈ sqrt(d^N * D^(2N))
end

@testset "negative squared quantities reject non-roundoff values" begin
    @test PMPSLab._nonnegative_squared(-eps(Float64), 1.0, "test quantity") == 0.0
    @test_throws DomainError PMPSLab._nonnegative_squared(-1.0, 1.0, "test quantity")
end

@testset "norm — two-physical-leg tensors" begin
    N, d, D = 2, 2, 2
    mps = rand(PeriodicMPS, ComplexF64, N, (d, d), D)
    @test norm(mps)^2 ≈ expectation(PeriodicTransferMatrix(mps))
    @test norm(mps)^2 ≈ real(overlap(mps, mps)) rtol=1e-10
end

@testset "normalize — returns a unit-norm copy" begin
    mps = rand(PeriodicMPS, ComplexF64, 4, 2, 3)
    original_norm = norm(mps)
    normalized = normalize(mps)
    site_scale = exp(log(original_norm) / length(mps))

    @test Base.isexported(PMPSLab, :normalize)
    @test normalized !== mps
    @test norm(normalized) ≈ 1
    @test norm(mps) ≈ original_norm
    @test all(
        normalized[site] ≈ mps[site] / site_scale
        for site in eachindex(mps)
    )
    dense_state = dense(mps)
    @test dense(normalized) ≈ dense_state / norm(dense_state)
    renormalized = normalize(normalized)
    @test dense(renormalized) ≈ dense(normalized)
    @test all(
        renormalized[site] ≈ normalized[site]
        for site in eachindex(normalized)
    )

    two_leg_mps = rand(PeriodicMPS, ComplexF64, 3, (2, 2), 2)
    dense_two_leg_state = dense(two_leg_mps)
    @test dense(normalize(two_leg_mps)) ≈
          dense_two_leg_state / norm(dense_two_leg_state)
end

@testset "normalize! — updates and returns its MPS" begin
    mps = rand(PeriodicMPS, ComplexF64, 4, 2, 3)
    tensor_objects = collect(mps)
    original_tensors = copy.(mps.tensors)
    original_dense = dense(mps)
    original_norm = norm(original_dense)
    site_scale = exp(log(original_norm) / length(mps))

    @test Base.isexported(PMPSLab, :normalize!)
    @test normalize!(mps) === mps
    @test all(mps[site] === tensor_objects[site] for site in eachindex(mps))
    @test norm(mps) ≈ 1
    @test dense(mps) ≈ original_dense / original_norm
    @test all(
        mps[site] ≈ original_tensors[site] / site_scale
        for site in eachindex(mps)
    )

    once_normalized = copy.(mps.tensors)
    @test normalize!(mps) === mps
    @test all(mps[site] === tensor_objects[site] for site in eachindex(mps))
    @test all(
        mps[site] ≈ once_normalized[site]
        for site in eachindex(mps)
    )
end

@testset "normalize — one-site and Z2 states" begin
    one_site = rand(PeriodicMPS, ComplexF64, 1, 3, 2)
    @test norm(normalize(one_site)) ≈ 1
    @test normalize!(one_site) === one_site
    @test norm(one_site) ≈ 1

    virtual_space = Z2Space(0 => 2, 1 => 1)
    physical_space = Z2Space(0 => 2)
    graded = rand(PeriodicMPS, ComplexF64, 3, physical_space, virtual_space)
    dense_graded = dense(graded)
    normalized = normalize(graded)

    @test norm(normalized) ≈ 1
    @test dense(normalized) ≈ dense_graded / norm(dense_graded)

    tensor_objects = collect(graded)
    @test normalize!(graded) === graded
    @test all(graded[site] === tensor_objects[site] for site in eachindex(graded))
    @test norm(graded) ≈ 1
end

@testset "normalize! — rejects zero states without mutation" begin
    mps = rand(PeriodicMPS, ComplexF64, 3, 2, 2)
    mps[1] = zero(mps[1])
    tensor_objects = collect(mps)
    tensor_values = copy.(mps.tensors)

    @test_throws ArgumentError normalize(mps)
    @test_throws ArgumentError normalize!(mps)
    @test all(mps[site] === tensor_objects[site] for site in eachindex(mps))
    @test all(mps[site] == tensor_values[site] for site in eachindex(mps))
end

@testset "normalize — avoids exponential overflow" begin
    site_count, physical_dimension, bond_dimension = 100, 400, 2
    mps = one_mps(
        ComplexF64, site_count, physical_dimension, bond_dimension
    )
    normalized = normalize(mps)
    site_scale = sqrt(physical_dimension) * bond_dimension

    @test isinf(squared_norm(mps))
    @test norm(mps) ≈ site_scale^site_count
    @test norm(normalized) ≈ 1
    @test all(
        normalized[site] ≈ mps[site] / site_scale
        for site in eachindex(mps)
    )

    in_place = one_mps(
        ComplexF64, site_count, physical_dimension, bond_dimension
    )
    tensor_objects = collect(in_place)
    @test normalize!(in_place) === in_place
    @test all(
        in_place[site] === tensor_objects[site]
        for site in eachindex(in_place)
    )
    @test norm(in_place) ≈ 1
end

@testset "squared_distance — zero for identical, positive for distinct" begin
    N, d, D = 3, 2, 2
    mps1 = rand(PeriodicMPS, ComplexF64, N, d, D)
    mps2 = rand(PeriodicMPS, ComplexF64, N, d, D)

    @test squared_distance(mps1, mps1) ≈ 0 atol=1e-12
    @test squared_distance(mps1, mps2) > 0
    @test squared_distance(mps1, mps2) ≈ squared_distance(mps2, mps1)
end

@testset "isapprox — gauge-equivalent states tolerate contraction roundoff" begin
    Random.seed!(310)
    mps = rand(PeriodicMPS, ComplexF64, 4, 2, 3)
    gauged = PeriodicMPS(copy.(mps.tensors))
    gaugefix!(gauged, ProjectorGaugeFix(; tol=1e-12, maxiter=100))

    @test squared_distance(mps, gauged) >= 0
    @test isapprox(mps, gauged)

    different = deepcopy(mps)
    different[1] = 1.001 * different[1]
    @test !isapprox(mps, different)
end

@testset "squared_distance — with Z2-graded spaces" begin
    N = 3
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mps1 = rand(PeriodicMPS, ComplexF64, N, P, V)
    mps2 = rand(PeriodicMPS, ComplexF64, N, P, V)

    @test squared_distance(mps1, mps1) ≈ 0 atol=1e-12
    @test squared_distance(mps1, mps2) > 0
    @test squared_distance(mps1, mps2) ≈ squared_distance(mps2, mps1)

    d12 = norm(dense(mps1) - dense(mps2))
    @test d12^2 ≈ squared_distance(mps1, mps2) rtol=1e-10
end
