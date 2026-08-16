_split_pending_iteration(run::SplitRun) = run.pending_iteration
_split_pending_iteration(saved::SplitSnapshot) = saved.pending_iteration

function _looptnr_position(
        completed_rg::Int,
        phase::LoopTNRPhase,
        active_split::Union{Nothing,SplitRun,SplitSnapshot},
        algorithm::LoopTNR,
    )
    max_rg_steps = algorithm.max_rg_steps
    if phase == TNR_INITIALIZE
        return "RG 0/$max_rg_steps normalization pending"
    elseif phase == TNR_PREPARE_SPLIT
        return "RG $(completed_rg + 1)/$max_rg_steps split pending"
    elseif phase == TNR_OPTIMIZE
        active_split === nothing &&
            return "RG $(completed_rg + 1)/$max_rg_steps split pending"
        pending = _split_pending_iteration(active_split)
        pending === nothing &&
            return "RG $(completed_rg + 1)/$max_rg_steps coarse graining pending"
        state = first(pending)
        site_count = length(sweep_domain(algorithm.split.sweep))
        return "RG $(completed_rg + 1)/$max_rg_steps · " *
            "split sweep $(state.completed_sweeps + 1) · next site $(state.site)/$site_count"
    elseif phase == TNR_COARSE_GRAIN
        return "RG $(completed_rg + 1)/$max_rg_steps coarse graining pending"
    end
    return "RG $completed_rg/$max_rg_steps complete"
end

_looptnr_position(run::LoopTNRRun) = _looptnr_position(
    run.completed_rg, run.phase, run.active_split, run.algorithm
)

_looptnr_position(saved::LoopTNRSnapshot) = _looptnr_position(
    saved.completed_rg,
    saved.phase,
    saved.active_split,
    saved.identity.algorithm,
)

function Base.show(io::IO, run::LoopTNRRun)
    print(io, "LoopTNRRun(", _looptnr_position(run), ")")
    return nothing
end

function Base.show(
        io::IO,
        session::LocalRunSession{R},
    ) where {R<:LoopTNRRun}
    print(
        io,
        "LocalRunSession(LoopTNRRun; status=", run_status(session),
        "; ", _looptnr_position(session.run),
        "; threaded=", session.threaded, ")",
    )
    return nothing
end

Base.show(
    io::IO,
    ::MIME"text/plain",
    session::LocalRunSession{R},
) where {R<:LoopTNRRun} = show(io, session)

function Base.show(io::IO, ::MIME"text/plain", run::LoopTNRRun)
    show(io, run)
    get(io, :compact, false)::Bool && return nothing

    print(
        io,
        "\n  committed cell: ", summary(run.cell),
        "\n  RG logs: ", length(run.log.rg_logs),
        "\n  measurements: ", length(run.measurement_results),
    )
    if !isnothing(run.active_split)
        active_split = run.active_split
        print(
            io,
            "\n  active split: ", length(active_split.log.site_logs),
            "/", length(active_split.algorithm.sweep), " site updates",
        )
    end
    return nothing
end

_result_completed_rg(result::LoopTNRResult) = last(result.log.rg_logs).rg

function Base.show(io::IO, result::LoopTNRResult)
    print(
        io,
        "LoopTNRResult(RG ", _result_completed_rg(result),
        "; ", length(result.measurement_results), " measurements; ",
        nameof(typeof(result.cell)), ")",
    )
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", result::LoopTNRResult)
    show(io, result)
    get(io, :compact, false)::Bool && return nothing

    print(
        io,
        "\n  cell: ", summary(result.cell),
        "\n  completed RG: ", _result_completed_rg(result),
        "\n  RG logs: ", length(result.log.rg_logs),
        "\n  measurements: ", length(result.measurement_results),
    )
    return nothing
end

function Base.show(io::IO, saved::LoopTNRSnapshot)
    print(
        io,
        "LoopTNRSnapshot(", _looptnr_position(saved),
        "; ", length(saved.measurement_results), " measurements)",
    )
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", saved::LoopTNRSnapshot)
    show(io, saved)
    get(io, :compact, false)::Bool && return nothing

    print(
        io,
        "\n  committed cell: ", summary(saved.cell),
        "\n  RG logs: ", length(saved.log.rg_logs),
        "\n  measurements: ", length(saved.measurement_results),
    )
    if !isnothing(saved.active_split)
        print(
            io,
            "\n  active split snapshot: ",
            length(saved.active_split.log.site_logs), "/",
            length(saved.identity.algorithm.split.sweep), " site updates",
        )
    end
    return nothing
end
