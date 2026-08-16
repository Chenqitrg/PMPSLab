"""
    RGLog

Control and timing information for one committed LoopTNR RG boundary. Physical
measurement results are stored separately in [`MeasurementHistory`](@ref).
"""
struct RGLog{S<:Union{Nothing,SplitLog}}
    rg::Int
    split_log::S
    coarse_grain_duration::Float64
end

function RGLog(
        rg::Integer,
        split_log::Union{Nothing,SplitLog},
        coarse_grain_duration::Real,
    )
    return RGLog(
        Int(rg),
        split_log,
        Float64(coarse_grain_duration),
    )
end

mutable struct LoopTNRLog <: ConvergenceLog
    rg_logs::Vector{Union{RGLog{Nothing},RGLog{SplitLog}}}
end

LoopTNRLog() = LoopTNRLog(Union{RGLog{Nothing},RGLog{SplitLog}}[])
