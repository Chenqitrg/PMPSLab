"""
    RGLog

Control and timing information for one committed LoopTNR RG boundary. Physical
measurement results are stored separately in [`MeasurementHistory`](@ref).
"""
struct RGLog{S}
    rg::Int
    split_log::S
    coarse_grain_time::Float64
end

function RGLog(
        rg::Integer,
        split_log,
        coarse_grain_time::Real,
    )
    return RGLog(
        Int(rg),
        split_log,
        Float64(coarse_grain_time),
    )
end

mutable struct LoopTNRLog <: ConvergenceLog
    rg_logs::Vector{RGLog}
end

LoopTNRLog() = LoopTNRLog(RGLog[])
