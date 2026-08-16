"""
    LoopTNR(; split, max_rg_steps, measurement=TNRMeasurement())

Configuration for Loop tensor-network renormalization.
"""
@kwdef struct LoopTNR{
        S<:ALSSplit,
        M<:TNRMeasurement,
    } <: AbstractAlgorithmConfig
    split::S
    max_rg_steps::Int
    measurement::M = TNRMeasurement()

    function LoopTNR(
            split::S,
            max_rg_steps::Integer,
            measurement::M,
        ) where {S<:ALSSplit,M<:TNRMeasurement}
        rg_count = Int(max_rg_steps)
        rg_count >= 0 || throw(ArgumentError(
            "max_rg_steps must be non-negative, got $max_rg_steps"
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
    default_looptnr(; max_rg_steps, dcut)

Build the standard dense-environment [`LoopTNR`](@ref) configuration. The
preliminary SVD retains at most `2dcut` states and periodic projector gauge
fixing then reduces the universal bond dimension to `dcut`.
"""
function default_looptnr(; max_rg_steps::Integer, dcut::Integer)
    dcut > 0 || throw(ArgumentError(
        "dcut must be positive, got $dcut"
    ))
    preliminary_rank = Base.checked_mul(2, Int(dcut))
    numerical_truncation = truncerror(; rtol=1.0e-13)
    split = ALSSplit(;
        ansatz=PeriodicGaugeAnsatz(;
            initial_split=SVDAnsatz(;
                trunc=truncrank(preliminary_rank) & numerical_truncation,
            ),
            gauge_fix=ProjectorGaugeFix(;
                tol=1.0e-13,
                maxiter=100,
                trunc=truncrank(dcut) & numerical_truncation,
            ),
        ),
        sweep=CircularSweep(; site_count=8, max_sweeps=40),
        solver=LDLTSolve(),
        environments=SplitEnvironmentPolicy(;
            normal=DenseTransferPolicy(),
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
    return LoopTNR(; split, max_rg_steps, measurement)
end
