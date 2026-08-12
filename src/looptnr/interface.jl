"""
    LoopTNR(split; maximal_rg, measurement=TNRMeasurement())
    LoopTNR(maxrank; maximal_rg, measurement=TNRMeasurement())

Configuration for Loop tensor-network renormalization. The convenience
constructor builds the standard eight-site periodic gauge split. The current
execution implementation supports the independently useful `maximal_rg=0`
boundary; later RG transformations require the project-specific coarse-grain
contraction.
"""
struct LoopTNR{
        S<:PeriodicSplit,
        M<:TNRMeasurement,
    } <: AbstractAlgorithmConfig
    split::S
    maximal_rg::Int
    measurement::M

    function LoopTNR(
            split::S,
            maximal_rg::Integer,
            measurement::M,
        ) where {S<:PeriodicSplit,M<:TNRMeasurement}
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

function LoopTNR(
        split::PeriodicSplit;
        maximal_rg::Integer,
        measurement::TNRMeasurement = TNRMeasurement(),
    )
    return LoopTNR(split, maximal_rg, measurement)
end

function LoopTNR(
        maxrank::Integer;
        maximal_rg::Integer,
        measurement::TNRMeasurement = TNRMeasurement(),
    )
    maxrank > 0 || throw(ArgumentError(
        "maxrank must be positive, got $maxrank"
    ))
    svd_truncation = truncerror(; rtol=1.0e-13)
    projector_truncation =
        truncrank(maxrank) & truncerror(; rtol=1.0e-13)
    ansatz = PeriodicGaugeAnsatz(
        SVDAnsatz(svd_truncation),
        ProjectorCanonical(; trscheme=projector_truncation),
    )
    split = PeriodicSplit(
        ansatz,
        CircularSweep(8; maximal_sweep=40),
        CholeskySolve();
        stopping=CostConvergence(
            ; atol=0,
            rtol=1.0e-10,
            min_sweeps=5,
            consecutive_sweeps=2,
        ),
    )
    return LoopTNR(split, maximal_rg, measurement)
end
