using Test
using CategoryData
using HalfIntegers: half
using PMPSLab
using Random
using TensorKit

function raw_periodic_dense(tensors)
    outputs = numout(first(tensors))
    left = id(TensorKit.scalartype(first(tensors)),
              PMPSLab.left_virtualspace(first(tensors)))
    for tensor in tensors
        left_tmp = left * repartition(tensor, 1, outputs)
        left = repartition(left_tmp, numind(left_tmp) - 1, 1)
    end

    codomain_count = length(codomain(left))
    physical_codomain = ProductSpace(ntuple(
        i -> codomain(left)[i + 1], codomain_count - 1
    ))
    result = zeros(TensorKit.scalartype(first(tensors)), physical_codomain)
    TensorKit.planartrace!(
        result,
        left,
        (Tuple(2:codomain_count), ()),
        ((1,), (codomain_count + 1,)),
        1.0,
        0.0,
    )
    return result
end

@testset "Semion closed-MPS standardization" begin
    semion_category = CategoryData.PMFC{2, 1, 0, 1, 1, 1}
    sector_type = CategoryData.Object{semion_category}
    vacuum = sector_type(1)
    semion = sector_type(2)
    @test frobenius_schur_indicator(semion) == -1

    virtual = dual(Vect[sector_type](vacuum => 2, semion => 2))
    physical = Vect[sector_type](vacuum => 1, semion => 1)
    rng = Xoshiro(0x51e4_10)
    raw = [
        rand(rng, ComplexF64, virtual ⊗ physical ← virtual)
        for _ in 1:2
    ]
    raw_spaces = space.(raw)
    before = raw_periodic_dense(raw)

    # Tensor-level `flip` is not a gauge transformation for an anyonic
    # category with a nontrivial Frobenius–Schur indicator.
    directly_flipped = flip.(raw, Ref((1, 3)))
    @test !isapprox(raw_periodic_dense(directly_flipped), before)

    mps = PeriodicMPS(raw)
    @test space.(raw) == raw_spaces
    @test all(!isdual ∘ PMPSLab.left_virtualspace, mps)
    @test all(!isdual ∘ PMPSLab.right_virtualspace, mps)
    @test dense(mps) ≈ before
end

function nondual_u1_plaquette_tensors()
    west = U1Space(0 => 1, 1 => 1)
    south = U1Space(0 => 1, -1 => 1)
    gate = rand(ComplexF64, west ⊗ south ← south ⊗ west)
    cell = PlaquetteGates(UniformGates(gate); copy=false)
    tensors = [
        transpose(cell.northwest, ((4, 3, 1), (2,)); copy=false),
        transpose(cell.southwest, ((3, 1, 2), (4,)); copy=false),
        transpose(cell.southeast, ((1, 2, 4), (3,)); copy=false),
        transpose(cell.northeast, ((2, 4, 3), (1,)); copy=false),
    ]
    return (; cell, tensors)
end

@testset "virtual gauge multiplication" begin
    tensors = nondual_u1_plaquette_tensors().tensors
    left, right = tensors[1], tensors[2]
    bond = PMPSLab.right_virtualspace(left)
    @test isdual(bond)
    @test bond == PMPSLab.left_virtualspace(right)

    standard_bond = flip(bond)
    gauge = unitary(ComplexF64, bond, standard_bond)
    gauged_left = mul_right_virtual(left, gauge)
    gauged_right = mul_left_virtual(adjoint(gauge), right)

    @test PMPSLab.right_virtualspace(gauged_left) == standard_bond
    @test PMPSLab.left_virtualspace(gauged_right) == standard_bond
    @test !isdual(standard_bond)

    @plansor before[-1 -2 -3 -4 -5; -6] :=
        left[-1 -2 -3; 1] * right[1 -4 -5; -6]
    @plansor after[-1 -2 -3 -4 -5; -6] :=
        gauged_left[-1 -2 -3; 1] * gauged_right[1 -4 -5; -6]
    @test after ≈ before

    # Half-integer SU(2) has a nontrivial Frobenius–Schur indicator. The
    # standardization must therefore be realized by explicit bond unitaries,
    # not by applying `flip` directly to tensor data.
    su2_bond = dual(SU2Space(1 // 2 => 1))
    trivial = SU2Space(0 => 1)
    su2_left = rand(ComplexF64, trivial ⊗ su2_bond ← su2_bond)
    su2_right = rand(ComplexF64, su2_bond ⊗ trivial ← su2_bond)
    su2_standard = flip(su2_bond)
    su2_gauge = unitary(ComplexF64, su2_bond, su2_standard)
    gauged_su2_left = mul_right_virtual(su2_left, su2_gauge)
    gauged_su2_right =
        mul_left_virtual(adjoint(su2_gauge), su2_right)

    @plansor su2_before[-1 -2 -3; -4] :=
        su2_left[-1 -2; 1] * su2_right[1 -3; -4]
    @plansor su2_after[-1 -2 -3; -4] :=
        gauged_su2_left[-1 -2; 1] * gauged_su2_right[1 -3; -4]
    @test su2_after ≈ su2_before
end

@testset "insert_gauge! accepts independent bond matrices" begin
    old_bond = ℂ^3
    new_bond = ℂ^2
    physical = ℂ^2
    mps = PeriodicMPS([
        rand(ComplexF64, old_bond ⊗ physical ← old_bond)
        for _ in 1:2
    ])
    right_matrix = rand(ComplexF64, old_bond ← new_bond)
    left_matrix = rand(ComplexF64, new_bond ← old_bond)
    expected_left = mul_right_virtual(mps[1], right_matrix)
    expected_right = mul_left_virtual(left_matrix, mps[2])

    @test insert_gauge!(
        mps, half(3), right_matrix, left_matrix
    ) === mps
    @test mps[1] ≈ expected_left
    @test mps[2] ≈ expected_right
    @test mps[half(3)] == new_bond
    @test assertconsistency(mps) === nothing
end

@testset "PeriodicMPS construction standardizes virtual bonds" begin
    fixture = nondual_u1_plaquette_tensors()
    raw = fixture.tensors
    raw_spaces = space.(raw)
    raw_dense = raw_periodic_dense(raw)
    @test any(isdual ∘ PMPSLab.right_virtualspace, raw)

    mps = PeriodicMPS(raw)
    @test all(!isdual ∘ PMPSLab.left_virtualspace, mps)
    @test all(!isdual ∘ PMPSLab.right_virtualspace, mps)
    @test dense(mps) ≈ raw_dense
    @test space.(raw) == raw_spaces

    standardized = standardize(mps)
    @test standardized !== mps
    @test dense(standardized) ≈ dense(mps)

    position = half(3)
    bond = mps[position]
    gauge = unitary(rand(ComplexF64, bond ← bond))
    before_gauge = dense(mps)
    @test insert_gauge!(
        mps, position, gauge, adjoint(gauge)
    ) === mps
    @test mps[position] == bond
    @test dense(mps) ≈ before_gauge
    @test assertconsistency(mps) === nothing

    periodic_bond = position + length(mps)
    @test mps[periodic_bond] == bond
    before_periodic_gauge = dense(mps)
    @test insert_gauge!(
        mps, periodic_bond, gauge, adjoint(gauge)
    ) === mps
    @test dense(mps) ≈ before_periodic_gauge
    @test !applicable(
        insert_gauge!, mps, 1, gauge, adjoint(gauge)
    )
    @test !applicable(
        insert_gauge!, raw, position, gauge, adjoint(gauge)
    )
    @test !applicable(standardize, raw)

    cell_mps = PeriodicMPS(fixture.cell)
    @test dense(cell_mps) ≈ raw_dense

    dual_bond = dual(U1Space(0 => 1, 1 => 1))
    physical = U1Space(0 => 1)
    one_site_raw = [rand(
        ComplexF64,
        dual_bond ⊗ physical ← dual_bond,
    )]
    one_site_dense = raw_periodic_dense(one_site_raw)
    one_site_mps = PeriodicMPS(one_site_raw)
    @test !isdual(PMPSLab.left_virtualspace(only(one_site_mps)))
    @test !isdual(PMPSLab.right_virtualspace(only(one_site_mps)))
    @test dense(one_site_mps) ≈ one_site_dense
end
