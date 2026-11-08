# Run from PMPSLab: julia --project=. examples/optimization/08_gauge_initialization.jl
using PMPSLab: SVDInitialization, GaugeInitialization
using PMPSLab
using Random: seed!
using Test: @test

seed!(1234)
target = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, 2, 2))

# PMPSLab owns the gauge algorithm and its implementation.
gauge = PMPSLab.ProjectorGaugeFix()

# First write the two operations explicitly.
manual = initial_approximation(target, SVDInitialization())
PMPSLab.gaugefix!(manual, gauge)

# PMPSLab's GaugeInitialization composes those same two operations.
# It does not implement another gauge-fixing algorithm.
initialization = GaugeInitialization(
    initialization=SVDInitialization(),
    gauge_fix=gauge,
)
prepared = initial_approximation(target, initialization)
@test dense(prepared) ≈ dense(manual)
@test dense(prepared) ≈ dense(target)
@show length(prepared)

# Pass this initialization to ALSSplit(initialization=initialization, ...).
# Supplying an existing initial MPS instead skips initialization entirely.
nothing
