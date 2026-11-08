# Run from PMPSLab: julia --project=. examples/environments/08_paired_mps.jl
using PMPSLab: DenseTransferMatrix
# Advanced: contract a two-physical-leg target with pairs of single-leg sites.
using PMPSLab: PeriodicMPS, contract_pairs
using PMPSLab: PeriodicTransferMatrix, transfer_at, expectation, squared_norm
using Test: @test

# Merge four single-physical-leg sites into two sites with two physical legs each.
paired = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
target = contract_pairs(paired)
@test length(target) == 2

# Target site 2 corresponds to sites 3 and 4 of paired.
E = transfer_at(target, paired, 2)
@test E ≈ DenseTransferMatrix(target[2], paired[3], paired[4])
transfers = PeriodicTransferMatrix(target, paired)
@test length(transfers) == length(target)

# contract_pairs preserves the physical state, so this overlap is its norm squared.
@test expectation(target, paired) ≈ squared_norm(paired)
@test expectation(paired, target) ≈ conj(expectation(target, paired))
@show length(target) length(paired)
# This mixed-representation cache uses the target site numbering.
# A change to either paired tensor must invalidate its corresponding target site.
nothing
