# Small, dense CDL diagnostics for the journey example, not an optimization step.
using Configurations: @option
using TensorKit, LinearAlgebra
import MPSKit, ExperimentLab

@option struct CDLFixture <: ExperimentLab.AbstractAlgorithmConfig
    seed::Int = 716
    sites::Int = 4
    corner_dimension::Int = 2
    normalize::Bool = true
end

@option struct CDLDiagnostics <: ExperimentLab.AbstractMeasurement
    rank_rtol::Float64 = 1e-6
end

const CDL_CONFIGS = ExperimentLab.ConfigRegistry(
    "CDLJourney.Fixture" => CDLFixture,
    "CDLJourney.Diagnostics" => CDLDiagnostics,
)
ExperimentLab.config_registry(::Type{CDLFixture}) = CDL_CONFIGS
ExperimentLab.config_registry(::Type{CDLDiagnostics}) = CDL_CONFIGS

struct CDLResult <: ExperimentLab.AbstractMeasurementResult
    target_squared_norm::Float64
    relative_squared_error::Float64
    relative_singular_values::Vector{Vector{Float64}}
    effective_pair_ranks::Vector{Int}
    bond_dimensions::Vector{Int}
    max_second_ratio::Float64
    rank1_relative_change::Float64
    rank1_target_error::Float64
end

function pair_singular_values(tensor)
    _, S, _, _ = svd_trunc(tensor, MPSKit.DefaultAlgorithm(trunc=MPSKit.TruncationStrategy()))
    return sort!(reduce(vcat, [collect(block.diag) for (_, block) in blocks(S)]); rev=true)
end

function ExperimentLab.measure(definition::CDLDiagnostics, input::NamedTuple)
    target, mps = input.target, input.mps
    spectra = [pair_singular_values(isodd(i) ? mps[i] : repartition(mps[i], 1, 2))
               for i in eachindex(mps)]
    relative = [values / first(values) for values in spectra]
    ranks = [count(>(definition.rank_rtol), values) for values in relative]

    # A diagnostic on a COPY: project each inserted bond onto its leading local
    # singular channel. Never feed these projected tensors back to the optimizer.
    projected = deepcopy(mps)
    for site in 1:2:length(mps)
        U, S, V, _ = svd_trunc(mps[site],
            MPSKit.DefaultAlgorithm(trunc=MPSKit.TruncationStrategy(maxrank=1)))
        projected[site] = U * S * V
    end
    reference, actual, rank1 = dense(target), dense(mps), dense(projected)
    C = norm(reference)^2
    @info "Measured CDL internal channels" max_rank=maximum(ranks)
    return CDLResult(C, norm(actual - reference)^2 / C, relative, ranks,
        [dim(MPSKit.right_virtualspace(t)) for t in mps],
        maximum(length(values) > 1 ? values[2] : 0.0 for values in relative),
        norm(rank1 - actual)^2 / norm(actual)^2, norm(rank1 - reference)^2 / C)
end
