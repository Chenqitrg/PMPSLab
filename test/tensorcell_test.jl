using Test
using CategoryData
using PMPSLab
using Random
using TensorKit

function test_gate(; west=ℂ^2, south=ℂ^3, north=south, east=west)
    return randn(ComplexF64, west ⊗ south, north ⊗ east)
end

gate_spaces(gate) = (codomain(gate)..., domain(gate)...)
all_nondual(cell) = all(
    !isdual(space) for gate in cell for space in gate_spaces(gate)
)

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
    shared_matrix = PlaquetteGates(matrix; copy=false)
    from_entries = PlaquetteGates(
        northwest, northeast, southwest, southeast
    )

    @test size(from_matrix) == (2, 2)
    @test from_matrix.northwest !== northwest
    @test from_matrix.northeast !== northeast
    @test from_matrix.southwest !== southwest
    @test from_matrix.southeast !== southeast
    @test shared_matrix.northwest === northwest
    @test shared_matrix.northeast === northeast
    @test shared_matrix.southwest === southwest
    @test shared_matrix.southeast === southeast
    @test collect(from_matrix) == matrix
    @test collect(from_entries) == matrix
    @test shared_matrix[1, 1] === northwest
    @test shared_matrix[1, 2] === northeast
    @test shared_matrix[2, 1] === southwest
    @test shared_matrix[2, 2] === southeast
    @test shared_matrix[3, 4] === northeast
    @test shared_matrix[0, -1] === southwest
    @test assertconsistency(from_matrix) === nothing

    @test_throws DimensionMismatch PlaquetteGates(fill(northwest, 1, 2))
end

@testset "gate cells — copy preserves topology without aliases" begin
    gate = test_gate()
    uniform = UniformGates(gate)
    checkerboard = CheckerboardGates(gate, gate)
    plaquette = PlaquetteGates(fill(gate, 2, 2); copy=false)

    for cell in (uniform, checkerboard, plaquette)
        copied = copy(cell)

        @test typeof(copied) == typeof(cell)
        @test collect(copied) == collect(cell)
        @test all(copied[index] !== cell[index] for index in CartesianIndices(cell))
        @test all(
            copied[index].data !== cell[index].data
            for index in CartesianIndices(cell)
        )
        @test length(unique(objectid.(collect(copied)))) == length(copied)
    end
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

    @test PlaquetteGates(plaquette; copy=false) === plaquette
    @test PlaquetteGates(plaquette) !== plaquette
end

@testset "gate leg multiplication" begin
    west, south, north, east = ℂ^2, ℂ^3, ℂ^4, ℂ^5
    new_west, new_south, new_north, new_east = ℂ^6, ℂ^7, ℂ^8, ℂ^9
    gate = randn(ComplexF64, west ⊗ south, north ⊗ east)
    west_matrix = randn(ComplexF64, new_west ← west)
    south_matrix = randn(ComplexF64, new_south ← south)
    north_matrix = randn(ComplexF64, north ← new_north)
    east_matrix = randn(ComplexF64, east ← new_east)

    west_result = mul_west(west_matrix, gate)
    south_result = mul_south(south_matrix, gate)
    north_result = mul_north(gate, north_matrix)
    east_result = mul_east(gate, east_matrix)

    @test space(west_result) == (new_west ⊗ south ← north ⊗ east)
    @test space(south_result) == (west ⊗ new_south ← north ⊗ east)
    @test space(north_result) == (west ⊗ south ← new_north ⊗ east)
    @test space(east_result) == (west ⊗ south ← north ⊗ new_east)
    @test west_result ≈ (west_matrix ⊗ id(ComplexF64, south)) * gate
    @test south_result ≈ (id(ComplexF64, west) ⊗ south_matrix) * gate
    @test north_result ≈ gate * (north_matrix ⊗ id(ComplexF64, east))
    @test east_result ≈ gate * (id(ComplexF64, north) ⊗ east_matrix)
end

@testset "gate cells standardize every periodic bond" begin
    semion_category = CategoryData.PMFC{2, 1, 0, 1, 1, 1}
    sector_type = CategoryData.Object{semion_category}
    vacuum = sector_type(1)
    semion = sector_type(2)
    @test frobenius_schur_indicator(semion) == -1
    nondual = Vect[sector_type](vacuum => 2, semion => 2)
    dual_bond = dual(nondual)
    rng = Xoshiro(0xce11)

    raw_uniform = rand(
        rng, ComplexF64, dual_bond ⊗ dual_bond ← dual_bond ⊗ dual_bond
    )
    raw_uniform_space = space(raw_uniform)
    uniform = UniformGates(raw_uniform)

    @test space(raw_uniform) == raw_uniform_space
    @test all_nondual(uniform)
    @test norm(uniform.gate) ≈ norm(raw_uniform)
    @test assertconsistency(uniform) === nothing

    # Only the black-east/white-west bond is dual. Its two explicit gauge
    # matrices must cancel even for Semion, whose FS indicator is nontrivial.
    trivial = Vect[sector_type](vacuum => 1)
    raw_black = rand(
        rng, ComplexF64, nondual ⊗ trivial ← trivial ⊗ dual_bond
    )
    raw_white = rand(
        rng, ComplexF64, dual_bond ⊗ trivial ← trivial ⊗ nondual
    )
    raw_spaces = (space(raw_black), space(raw_white))
    checkerboard = CheckerboardGates(raw_black, raw_white)

    @plansor before[-1 -2 -4; -3 -5 -6] :=
        raw_black[-1 -2; -3 1] * raw_white[1 -4; -5 -6]
    @plansor after[-1 -2 -4; -3 -5 -6] :=
        checkerboard.black[-1 -2; -3 1] *
        checkerboard.white[1 -4; -5 -6]

    @test space(raw_black) == raw_spaces[1]
    @test space(raw_white) == raw_spaces[2]
    @test all_nondual(checkerboard)
    @test after ≈ before

    raw_upper = rand(
        rng, ComplexF64, trivial ⊗ dual_bond ← nondual ⊗ trivial
    )
    raw_lower = rand(
        rng, ComplexF64, trivial ⊗ nondual ← dual_bond ⊗ trivial
    )
    vertical = CheckerboardGates(raw_upper, raw_lower)
    @plansor vertical_before[-1 -4 -5; -2 -3 -6] :=
        raw_upper[-1 1; -2 -3] * raw_lower[-4 -5; 1 -6]
    @plansor vertical_after[-1 -4 -5; -2 -3 -6] :=
        vertical.black[-1 1; -2 -3] *
        vertical.white[-4 -5; 1 -6]

    @test all_nondual(vertical)
    @test vertical_after ≈ vertical_before

    raw_plaquette = [
        rand(
            rng,
            ComplexF64,
            dual_bond ⊗ dual_bond ← dual_bond ⊗ dual_bond,
        ) for _ in 1:2, _ in 1:2
    ]
    raw_plaquette_spaces = space.(raw_plaquette)
    plaquette = PlaquetteGates(raw_plaquette; copy=false)

    @test space.(raw_plaquette) == raw_plaquette_spaces
    @test all_nondual(plaquette)
    @test norm.(collect(plaquette)) ≈ norm.(raw_plaquette)
    @test assertconsistency(plaquette) === nothing

    for cell in (uniform, checkerboard, plaquette)
        standardized = standardize(cell)
        @test standardized !== cell
        @test typeof(standardized) == typeof(cell)
        @test all_nondual(standardized)
        @test all(
            standardized[index] ≈ cell[index]
            for index in CartesianIndices(cell)
        )
    end
end

@testset "gate cells convert to U-shaped periodic MPS" begin
    horizontal_ab = ℂ^2
    horizontal_ba = ℂ^3
    horizontal_cd = ℂ^4
    horizontal_dc = ℂ^5
    vertical_ac = ℂ^6
    vertical_ca = ℂ^7
    vertical_bd = ℂ^8
    vertical_db = ℂ^9

    northwest = randn(
        ComplexF64,
        horizontal_ba ⊗ vertical_ac,
        vertical_ca ⊗ horizontal_ab,
    )
    northeast = randn(
        ComplexF64,
        horizontal_ab ⊗ vertical_bd,
        vertical_db ⊗ horizontal_ba,
    )
    southwest = randn(
        ComplexF64,
        horizontal_dc ⊗ vertical_ca,
        vertical_ac ⊗ horizontal_cd,
    )
    southeast = randn(
        ComplexF64,
        horizontal_cd ⊗ vertical_db,
        vertical_bd ⊗ horizontal_dc,
    )
    plaquette = PlaquetteGates(
        northwest, northeast, southwest, southeast
    )
    mps = PeriodicMPS(plaquette)

    @test length(mps) == 4
    @test assertconsistency(mps) === nothing
    @test all(
        !isdual(PMPSLab.left_virtualspace(tensor)) &&
        !isdual(PMPSLab.right_virtualspace(tensor))
        for tensor in mps
    )

    uniform = UniformGates(test_gate())
    checkerboard = CheckerboardGates(uniform)
    for cell in (uniform, checkerboard)
        promoted = PeriodicMPS(PlaquetteGates(cell; copy=false))
        @test all(PeriodicMPS(cell)[site] ≈ promoted[site] for site in 1:4)
    end
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
