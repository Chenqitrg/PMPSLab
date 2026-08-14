# Replicate a finite unit cell or lift it to an infinite MPS.

"""
    lift(mps::PeriodicMPS, fold::Int) -> PeriodicMPS

Replicate the unit cell `fold` times by concatenation.
Corresponds to lifting a periodic cell to a finite cover.
"""
lift(mps::PeriodicMPS, fold::Int) = repeat(mps, fold)

"""
    lift(mps::PeriodicMPS, ::Val{Inf}) -> InfiniteMPS

Lift to an infinite MPS (MPSKit.InfiniteMPS) by repeating the unit cell.
"""
lift(mps::PeriodicMPS, ::Val{Inf}) = MPSKit.InfiniteMPS(collect(mps))
