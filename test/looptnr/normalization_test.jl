using LinearAlgebra
using TensorKit

function normalization_gate(::Type{T}=ComplexF64) where {T<:Number}
    return randn(T, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2)
end

function set_gate_norm!(gate, target::Real)
    rmul!(gate, target / norm(gate, Inf))
    return gate
end

independent_gate_norms(cell::UniformGates) = [norm(cell.gate, Inf)]
independent_gate_norms(cell::CheckerboardGates) =
    [norm(cell.black, Inf), norm(cell.white, Inf)]
independent_gate_norms(cell::PlaquetteGates) = [
    norm(cell.northwest, Inf),
    norm(cell.northeast, Inf),
    norm(cell.southwest, Inf),
    norm(cell.southeast, Inf),
]

@testset "TensorCell infinity normalization — cell variants" begin
    normalization = InfinityNormNormalization()
    cells = (
        UniformGates(set_gate_norm!(normalization_gate(Float64), 8.0)),
        CheckerboardGates(
            set_gate_norm!(normalization_gate(), 4.0),
            set_gate_norm!(normalization_gate(), 9.0),
        ),
        PlaquetteGates(
            set_gate_norm!(normalization_gate(), 1.0),
            set_gate_norm!(normalization_gate(), 4.0),
            set_gate_norm!(normalization_gate(), 16.0),
            set_gate_norm!(normalization_gate(), 64.0),
        ),
    )
    expected_scales = (8.0, 6.0, 8.0)

    for (cell, expected_scale) in zip(cells, expected_scales)
        gates_before = collect(cell)
        identities_before = objectid.(gates_before)
        result = normalize!(cell, normalization)

        @test result isa NormalizationResult
        @test result.scale ≈ expected_scale
        @test result.log_scale ≈ log(expected_scale)
        @test objectid.(collect(cell)) == identities_before
        @test exp(sum(log, independent_gate_norms(cell)) /
                  length(independent_gate_norms(cell))) ≈ 1
    end
end

@testset "TensorCell infinity normalization — aliases scale once" begin
    gate = set_gate_norm!(normalization_gate(), 4.0)
    checkerboard = CheckerboardGates(gate, gate)

    result = normalize!(checkerboard, InfinityNormNormalization())

    @test result.scale ≈ 4
    @test checkerboard.black === gate
    @test checkerboard.white === gate
    @test norm(gate, Inf) ≈ 1

    plaquette_gate = set_gate_norm!(normalization_gate(), 7.0)
    plaquette = PlaquetteGates(fill(plaquette_gate, 2, 2); copy=false)
    normalize!(plaquette, InfinityNormNormalization())
    @test norm(plaquette_gate, Inf) ≈ 1

    repeated = set_gate_norm!(normalization_gate(), 4.0)
    distinct = set_gate_norm!(normalization_gate(), 9.0)
    partially_aliased = PlaquetteGates(
        repeated, repeated, repeated, distinct
    )
    partial_result = normalize!(
        partially_aliased, InfinityNormNormalization()
    )
    @test partial_result.scale ≈ 6
    @test norm(repeated, Inf) ≈ 2 / 3
    @test norm(distinct, Inf) ≈ 3 / 2
end

@testset "TensorCell infinity normalization — logarithmic extremes" begin
    large = set_gate_norm!(normalization_gate(), 1.0e300)
    moderate = set_gate_norm!(normalization_gate(), 1.0e100)
    cell = CheckerboardGates(large, moderate)

    result = normalize!(cell, InfinityNormNormalization())

    @test isfinite(result.scale)
    @test result.scale ≈ 1.0e200 rtol=1.0e-12
    @test result.log_scale ≈ log(1.0e200)
    @test norm(cell.black, Inf) ≈ 1.0e100 rtol=1.0e-12
    @test norm(cell.white, Inf) ≈ 1.0e-100 rtol=1.0e-12

    small = set_gate_norm!(normalization_gate(), 1.0e-300)
    less_small = set_gate_norm!(normalization_gate(), 1.0e-100)
    small_cell = CheckerboardGates(small, less_small)
    small_result = normalize!(small_cell, InfinityNormNormalization())
    @test small_result.scale ≈ 1.0e-200 rtol=1.0e-12
    @test exp(sum(log, independent_gate_norms(small_cell)) / 2) ≈ 1
end

@testset "TensorCell infinity normalization — failures are atomic" begin
    normalization = InfinityNormNormalization()

    zero_gate = normalization_gate()
    rmul!(zero_gate, 0)
    finite_gate = normalization_gate()
    zero_cell = CheckerboardGates(zero_gate, finite_gate)
    zero_before = copy.(collect(zero_cell))
    @test_throws ArgumentError normalize!(zero_cell, normalization)
    @test all(collect(zero_cell) .== zero_before)

    finite_gate = normalization_gate()
    infinite_gate = normalization_gate()
    infinite_gate.data[begin] = Inf
    infinite_cell = CheckerboardGates(finite_gate, infinite_gate)
    infinite_before = copy.(collect(infinite_cell))
    @test_throws DomainError normalize!(infinite_cell, normalization)
    @test all(collect(infinite_cell) .== infinite_before)

    finite_gate = normalization_gate()
    nan_gate = normalization_gate()
    nan_gate.data[begin] = NaN
    nan_cell = CheckerboardGates(finite_gate, nan_gate)
    finite_before = copy(finite_gate)
    @test_throws DomainError normalize!(nan_cell, normalization)
    @test finite_gate == finite_before
    @test isnan(nan_gate.data[begin])
end
