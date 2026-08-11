using Test
using PMPSLab
using Random
using TensorKit

function test_gate(; west=ℂ^2, south=ℂ^3, north=south, east=west)
    return randn(ComplexF64, west ⊗ south, north ⊗ east)
end

@testset "UniformGates — periodic 1×1 indexing" begin
    gate = test_gate()
    cell = UniformGates(gate)

    @test cell isa AbstractTensorCell
    @test cell isa AbstractMatrix
    @test size(cell) == (1, 1)
    @test axes(cell) == (Base.OneTo(1), Base.OneTo(1))
    @test length(cell) == 1
    @test cell[1, 1] === gate
    @test cell[-3, 4] === gate
    @test only(cell) === gate
    @test assertconsistency(cell) === nothing
end

@testset "CheckerboardGates — staggered periodic indexing" begin
    black = test_gate(west=ℂ^2, south=ℂ^3, north=ℂ^4, east=ℂ^5)
    white = test_gate(west=ℂ^5, south=ℂ^4, north=ℂ^3, east=ℂ^2)
    cell = CheckerboardGates(black, white)

    @test size(cell) == (1, 2)
    @test cell[1, 1] === black
    @test cell[1, 2] === white
    @test cell[2, 1] === white
    @test cell[2, 2] === black
    @test cell[3, 3] === black
    @test cell[0, 0] === black
    @test cell[-1, -1] === black
    @test collect(cell) == [black white]
    @test assertconsistency(cell) === nothing
end

@testset "PlaquetteGates — periodic 2×2 indexing" begin
    northwest = test_gate()
    northeast = test_gate()
    southwest = test_gate()
    southeast = test_gate()
    matrix = [northwest northeast; southwest southeast]

    from_matrix = PlaquetteGates(matrix)
    from_entries = PlaquetteGates(
        northwest, northeast, southwest, southeast
    )

    @test size(from_matrix) == (2, 2)
    @test from_matrix.northwest === northwest
    @test from_matrix.northeast === northeast
    @test from_matrix.southwest === southwest
    @test from_matrix.southeast === southeast
    @test collect(from_matrix) == matrix
    @test collect(from_entries) == matrix
    @test from_matrix[1, 1] === northwest
    @test from_matrix[1, 2] === northeast
    @test from_matrix[2, 1] === southwest
    @test from_matrix[2, 2] === southeast
    @test from_matrix[3, 4] === northeast
    @test from_matrix[0, -1] === southwest
    @test assertconsistency(from_matrix) === nothing

    @test_throws DimensionMismatch PlaquetteGates(fill(northwest, 1, 2))
end

@testset "gate cells promote to larger unit cells" begin
    gate = test_gate()
    uniform = UniformGates(gate)
    checkerboard = CheckerboardGates(uniform)
    uniform_plaquette = PlaquetteGates(uniform)

    @test checkerboard.black !== gate
    @test checkerboard.white !== gate
    @test checkerboard.black !== checkerboard.white
    @test checkerboard.black ≈ gate
    @test checkerboard.white ≈ gate
    @test all(entry !== gate for entry in uniform_plaquette)
    @test all(entry ≈ gate for entry in uniform_plaquette)
    @test length(unique(objectid.(collect(uniform_plaquette)))) == 4

    shared_checkerboard = CheckerboardGates(uniform; copy=false)
    shared_uniform_plaquette = PlaquetteGates(uniform; copy=false)

    @test shared_checkerboard.black === gate
    @test shared_checkerboard.white === gate
    @test all(entry === gate for entry in shared_uniform_plaquette)

    black = test_gate()
    white = test_gate()
    checkerboard = CheckerboardGates(black, white)
    plaquette = PlaquetteGates(checkerboard)

    @test plaquette.northwest !== black
    @test plaquette.northeast !== white
    @test plaquette.southwest !== white
    @test plaquette.southeast !== black
    @test plaquette.northwest !== plaquette.southeast
    @test plaquette.northeast !== plaquette.southwest
    @test collect(plaquette) == [black white; white black]

    shared_plaquette = PlaquetteGates(checkerboard; copy=false)

    @test shared_plaquette.northwest === black
    @test shared_plaquette.northeast === white
    @test shared_plaquette.southwest === white
    @test shared_plaquette.southeast === black
end

@testset "gate cells reject open periodic bonds" begin
    open_uniform = test_gate(east=ℂ^4)
    @test_throws TensorKit.SpaceMismatch UniformGates(open_uniform)

    black = test_gate(west=ℂ^2, south=ℂ^3, north=ℂ^4, east=ℂ^5)
    wrong_black_neighbor = test_gate(
        west=ℂ^6, south=ℂ^4, north=ℂ^3, east=ℂ^2
    )
    wrong_white_neighbor = test_gate(
        west=ℂ^5, south=ℂ^4, north=ℂ^3, east=ℂ^6
    )
    @test_throws TensorKit.SpaceMismatch CheckerboardGates(
        black, wrong_black_neighbor
    )
    @test_throws TensorKit.SpaceMismatch CheckerboardGates(
        black, wrong_white_neighbor
    )

    compatible = test_gate()
    incompatible_northeast = test_gate(west=ℂ^4, east=ℂ^4)
    plaquette = [compatible incompatible_northeast; compatible compatible]
    @test_throws TensorKit.SpaceMismatch PlaquetteGates(plaquette)
end

@testset "gate cells — random generation" begin
    uniform_space = ℂ^2 ⊗ ℂ^3 ← ℂ^3 ⊗ ℂ^2
    uniform = rand(UniformGates, ComplexF64, uniform_space)
    seeded_uniform = rand(Xoshiro(1), UniformGates, Float64, uniform_space)
    replayed_uniform = rand(Xoshiro(1), UniformGates, Float64, uniform_space)

    @test eltype(uniform.gate) == ComplexF64
    @test space(uniform.gate) == uniform_space
    @test seeded_uniform.gate == replayed_uniform.gate

    blackspace = ℂ^2 ⊗ ℂ^3 ← ℂ^4 ⊗ ℂ^5
    checkerboard = rand(
        Xoshiro(2), CheckerboardGates, Float64, blackspace
    )

    @test eltype(checkerboard.black) == Float64
    @test space(checkerboard.black) == blackspace
    @test codomain(checkerboard.white) == reverse(domain(blackspace))
    @test domain(checkerboard.white) == reverse(codomain(blackspace))
    @test checkerboard.black !== checkerboard.white

    horizontal_ab = ℂ^2
    horizontal_ba = ℂ^3
    horizontal_cd = ℂ^4
    horizontal_dc = ℂ^5
    vertical_ac = ℂ^2
    vertical_ca = ℂ^3
    vertical_bd = ℂ^4
    vertical_db = ℂ^5
    northwest_space =
        horizontal_ba ⊗ vertical_ac ← vertical_ca ⊗ horizontal_ab
    northeast_space =
        horizontal_ab ⊗ vertical_bd ← vertical_db ⊗ horizontal_ba
    southwest_space =
        horizontal_dc ⊗ vertical_ca ← vertical_ac ⊗ horizontal_cd
    southeast_space =
        horizontal_cd ⊗ vertical_db ← vertical_bd ⊗ horizontal_dc
    plaquette_spaces = [
        northwest_space northeast_space; southwest_space southeast_space
    ]
    plaquette = rand(PlaquetteGates, ComplexF64, plaquette_spaces)

    @test eltype(first(plaquette)) == ComplexF64
    @test space.(collect(plaquette)) == plaquette_spaces
    @test_throws DimensionMismatch rand(
        PlaquetteGates, ComplexF64, fill(uniform_space, 1, 2)
    )
end

@testset "gate cells require square four-leg tensors" begin
    mps_tensor = randn(ComplexF64, ℂ^2 ⊗ ℂ^3, ℂ^4)
    @test_throws MethodError UniformGates(mps_tensor)
    @test_throws MethodError CheckerboardGates(mps_tensor, mps_tensor)
    @test_throws MethodError PlaquetteGates(fill(mps_tensor, 2, 2))
end

@testset "gate cells — plain-text display" begin
    gate = test_gate()
    uniform = UniformGates(gate)
    uniform_summary = "1×1 UniformGates(ComplexF64, ComplexSpace)"

    @test summary(uniform) == uniform_summary
    @test sprint(show, uniform) == uniform_summary
    @test sprint(
        show, MIME"text/plain"(), uniform; context=(:compact => true,)
    ) == uniform_summary

    uniform_shown = sprint(show, MIME"text/plain"(), uniform)
    @test startswith(uniform_shown, uniform_summary * ":\n")
    @test occursin("    ⋮\n    │\n…─[1,1]─…\n    │\n    ⋮", uniform_shown)
    @test occursin("[1,1] W=ℂ^2, S=ℂ^3, N=ℂ^3, E=ℂ^2", uniform_shown)
    @test !occursin("TensorMap", uniform_shown)

    checkerboard = CheckerboardGates(gate, gate)
    checkerboard_shown = sprint(show, MIME"text/plain"(), checkerboard)
    @test startswith(
        checkerboard_shown,
        "1×2 CheckerboardGates(ComplexF64, ComplexSpace):\n",
    )
    @test occursin("…─[1,1]─────[1,2]─…", checkerboard_shown)
    @test occursin("…─[2,1]─────[2,2]─…", checkerboard_shown)

    plaquette = PlaquetteGates(fill(gate, 2, 2))
    plaquette_shown = sprint(show, MIME"text/plain"(), plaquette)
    @test startswith(
        plaquette_shown,
        "2×2 PlaquetteGates(ComplexF64, ComplexSpace):\n",
    )
    @test occursin("…─[1,1]─────[1,2]─…", plaquette_shown)
    @test occursin("…─[2,1]─────[2,2]─…", plaquette_shown)
end
