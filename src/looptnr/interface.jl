"""
    LoopTNR(; split, maximal_rg, measurement=TNRMeasurement())

Configuration for Loop tensor-network renormalization.
"""
@kwdef struct LoopTNR{
        S<:ALSSplit,
        M<:TNRMeasurement,
    } <: AbstractAlgorithmConfig
    split::S
    maximal_rg::Int
    measurement::M = TNRMeasurement()

    function LoopTNR(
            split::S,
            maximal_rg::Integer,
            measurement::M,
        ) where {S<:ALSSplit,M<:TNRMeasurement}
        rg_count = Int(maximal_rg)
        rg_count >= 0 || throw(ArgumentError(
            "maximal_rg must be non-negative, got $maximal_rg"
        ))
        domain = sweep_domain(split.sweep)
        domain == Base.OneTo(8) || throw(DimensionMismatch(
            "LoopTNR split sweep domain $domain does not match the required " *
            "eight-site domain $(Base.OneTo(8))"
        ))
        return new{S,M}(split, rg_count, measurement)
    end
end

"""
    default_looptnr(; maximal_rg, dcut)

Build the standard dense-environment [`LoopTNR`](@ref) configuration. The
preliminary SVD retains at most `2dcut` states and periodic canonicalization
then reduces the universal bond dimension to `dcut`.
"""
function default_looptnr(; maximal_rg::Integer, dcut::Integer)
    dcut > 0 || throw(ArgumentError(
        "dcut must be positive, got $dcut"
    ))
    preliminary_rank = Base.checked_mul(2, Int(dcut))
    numerical_truncation = truncerror(; rtol=1.0e-13)
    split = ALSSplit(;
        ansatz=PeriodicGaugeAnsatz(;
            svd_pre=SVDAnsatz(;
                trunc=truncrank(preliminary_rank) & numerical_truncation,
            ),
            canonical=ProjectorCanonical(;
                tol=1.0e-13,
                maxiter=100,
                trunc=truncrank(dcut) & numerical_truncation,
            ),
        ),
        sweep=CircularSweep(; N=8, maximal_sweep=40),
        solve=LDLTSolve(),
        environments=SplitEnvironmentPolicy(;
            norm=DenseTransferPolicy(),
            mixed=DenseTransferPolicy(),
        ),
        stopping=CostTolerance(;
            atol=1.0e-9,
            rtol=1.0e-2,
            min_sweeps=5,
        ),
    )
    measurement = TNRMeasurement(;
        normalization=InfinityNormNormalization(),
        fixed_point=NoFixedPointMeasurement(),
    )
    return LoopTNR(; split, maximal_rg, measurement)
end
