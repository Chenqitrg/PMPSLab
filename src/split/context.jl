# ------------------------------------------------------------------------------
# SplitContext — state of one two-leg → paired single-leg split run
#
# The context owns mutable numerical working state but no sweep position.
# `update!` performs one site's update in place from the supplied `SweepState`.
# The sweep scheme supplies the visit order. Caches track which derived products
# remain valid and reject unsupported update/query sequences without performing
# implicit tensor contractions.
# ------------------------------------------------------------------------------

# Each two-physical-leg target tensor maps to two single-physical-leg
# approximation tensors.
struct SplitContext{
        E, S,
        T<:PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, A,
        NC, MC,
    } <: AlgorithmContext{E,S}
    target::T
    approximation::A
    normal_cache::NC
    mixed_cache::MC
    target_squared_norm::E
end

target_site_for(approximation_site::Int) = cld(approximation_site, 2)
current_approximation(ctx::SplitContext, site::Int) = ctx.approximation[site]
paired_approximation(ctx::SplitContext, site::Int) =
    iseven(site) ? ctx.approximation[site - 1] : ctx.approximation[site + 1]
target_tensor(ctx::SplitContext, approximation_site::Int) =
    ctx.target[target_site_for(approximation_site)]

function _build_split_context(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        approximation::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
        environments::SplitEnvironmentPolicy,
    ) where {E,S}
    normal_cache = build_cache(approximation, approximation, environments.normal)
    mixed_cache = build_cache(approximation, target, environments.mixed)
    target_squared_norm = expectation(target, target)
    return SplitContext(
        target,
        approximation,
        normal_cache,
        mixed_cache,
        target_squared_norm,
    )
end

SplitContext(
    target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
    approximation::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
) where {E,S} = _build_split_context(target, approximation, SplitEnvironmentPolicy())

function _solve_local_update(
        ctx::SplitContext, solver::AbstractSolve, state::SweepState
    )
    approximation_site = state.site
    1 <= approximation_site <= length(ctx.approximation) ||
        throw(BoundsError(ctx.approximation, approximation_site))
    target_site = target_site_for(approximation_site)
    normal_environment = environment(ctx.normal_cache, approximation_site)
    mixed_environment = environment(ctx.mixed_cache, target_site)
    target = target_tensor(ctx, approximation_site)
    paired = paired_approximation(ctx, approximation_site)'
    rhs = _split_rhs(
        mixed_environment, target, paired, approximation_site
    )
    initial = transpose(
        current_approximation(ctx, approximation_site), ((2,), (1, 3))
    )
    solution = _solve_split_norm(
        solver, initial, normal_environment, rhs
    )

    # Local-environment estimate of ‖target - approximation‖².
    cost = real(ctx.target_squared_norm - conj(tr(rhs * solution')))

    return transpose(solution, ((2, 1), (3,))), cost
end

_pair_completed(st::SweepState{Forward}) = iseven(st.site)
_pair_completed(st::SweepState{Backward}) = isodd(st.site)

"""
    update!(ctx::SplitContext, solver::AbstractSolve, st::SweepState)
    update!(ctx::SplitContext, solver::AbstractSolve, environments, st::SweepState)

One site's update + validated directional cache refresh. An incomplete B pair
only invalidates its mixed AB factor; completing the pair constructs that factor
once and advances the AB cache. Unsupported visit orders fail with
`StaleCacheError` instead of triggering extra contractions or returning stale
environments. The three-argument method uses the default dense transfer
policies; non-default policies are supplied explicitly and are never stored in
`SplitContext`.
"""
function update!(
        ctx::SplitContext,
        solver::AbstractSolve,
        environments::SplitEnvironmentPolicy,
        st::SweepState{D},
    ) where {D}
    new, cost = _solve_local_update(ctx, solver, st)
    ctx.approximation[st.site] = new

    if _requires_rebuild(ctx.normal_cache, st)
        if D === Forward
            rebuild_right!(
                ctx.normal_cache,
                ctx.approximation,
                ctx.approximation,
                environments.normal,
            )
            rebuild_right!(
                ctx.mixed_cache,
                ctx.approximation,
                ctx.target,
                environments.mixed,
            )
        else
            rebuild_left!(
                ctx.normal_cache,
                ctx.approximation,
                ctx.approximation,
                environments.normal,
            )
            rebuild_left!(
                ctx.mixed_cache,
                ctx.approximation,
                ctx.target,
                environments.mixed,
            )
        end
        return new, cost
    end

    normal_transfer = transfer_at(
        ctx.approximation,
        ctx.approximation,
        st.site,
        environments.normal,
    )
    advance!(ctx.normal_cache, normal_transfer, st)

    target_site = target_site_for(st.site)
    if _pair_completed(st)
        pair_state = SweepState{D}(target_site, st.completed_sweeps)
        mixed_transfer = transfer_at(
            ctx.approximation,
            ctx.target,
            target_site,
            environments.mixed,
        )
        advance!(ctx.mixed_cache, mixed_transfer, pair_state)
    else
        # Avoid constructing the mixed transfer until both approximation sites
        # in the pair are current.
        invalidate!(ctx.mixed_cache, target_site)
    end
    return new, cost
end

update!(
    ctx::SplitContext, solver::AbstractSolve, st::SweepState
) = update!(ctx, solver, SplitEnvironmentPolicy(), st)
