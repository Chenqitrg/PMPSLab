"""Solve one site and refresh affected caches using the workspace's configuration."""
function update!(state::SplitState, position::LatticeIteratorLab.SweepState)
    tensor, cost = _solve_local_update(state, position)
    state.approximation[position.site] = tensor
    _refresh_split_caches!(state, position)
    _update_regularization!(state.regularization, state, position.site)
    return tensor, cost
end

# Solve one local equation without mutating the MPS or its caches.

function _solve_local_update(
        ctx::SplitState, state::LatticeIteratorLab.SweepState
    )
    approximation_site = state.site
    1 <= approximation_site <= length(ctx.approximation) ||
        throw(BoundsError(ctx.approximation, approximation_site))
    target_site = target_site_for(approximation_site)
    normal_environment = PeriodicProductsLab.environment(ctx.normal_cache, approximation_site)
    mixed_environment = PeriodicProductsLab.environment(ctx.mixed_cache, target_site)
    target = ctx.target[target_site]
    paired = paired_approximation(ctx, approximation_site)'
    rhs = _split_rhs(
        mixed_environment, target, paired, approximation_site
    )
    initial = transpose(
        ctx.approximation[approximation_site], ((2,), (1, 3))
    )
    solution = _regularized_solution(
        ctx.regularization, ctx, approximation_site, initial, normal_environment, rhs,
    )

    # Local-environment estimate of ‖target - approximation‖².
    norm_action = _split_norm_action(solution, normal_environment)
    cost = real(ctx.target_squared_norm + tr(norm_action * solution') -
                2 * tr(rhs * solution'))

    return transpose(solution, ((2, 1), (3,))), cost / _cost_scale(ctx)
end

# Refresh derived environments after the approximation tensor has been replaced.
# This layer owns pair completion and directional cache validity, not the solve.

_pair_completed(st::LatticeIteratorLab.SweepState{LatticeIteratorLab.Forward}) = iseven(st.site)
_pair_completed(st::LatticeIteratorLab.SweepState{LatticeIteratorLab.Backward}) = isodd(st.site)

function _refresh_split_caches!(
        ctx::SplitState,
        st::LatticeIteratorLab.SweepState{D},
    ) where {D}
    environments = ctx.algorithm.environments
    if (D === LatticeIteratorLab.Forward && st.site == length(ctx.approximation)) ||
       (D === LatticeIteratorLab.Backward && st.site == 1)
        # Boundary steps skip incremental cache advances, so report both changed transfers here
        # before an incremental rebuild decides which cached products to reuse.
        PeriodicProductsLab.invalidate!(ctx.normal_cache, st.site)
        PeriodicProductsLab.invalidate!(ctx.mixed_cache, target_site_for(st.site))
        if D === LatticeIteratorLab.Forward
            rebuild_right_transfer_cache!(
                ctx.normal_cache,
                ctx.approximation,
                ctx.approximation,
                environments.normal,
            )
            rebuild_right_transfer_cache!(
                ctx.mixed_cache,
                ctx.approximation,
                ctx.target,
                environments.mixed,
            )
        else
            rebuild_left_transfer_cache!(
                ctx.normal_cache,
                ctx.approximation,
                ctx.approximation,
                environments.normal,
            )
            rebuild_left_transfer_cache!(
                ctx.mixed_cache,
                ctx.approximation,
                ctx.target,
                environments.mixed,
            )
        end
        return nothing
    end

    normal_transfer = transfer_at(
        ctx.approximation,
        ctx.approximation,
        st.site,
        environments.normal,
    )
    PeriodicProductsLab.advance!(ctx.normal_cache, normal_transfer, _cache_state(st))

    target_site = target_site_for(st.site)
    if _pair_completed(st)
        mixed_transfer = transfer_at(
            ctx.approximation,
            ctx.target,
            target_site,
            environments.mixed,
        )
        PeriodicProductsLab.advance!(ctx.mixed_cache, mixed_transfer, _cache_state(st, target_site))
    else
        # Avoid constructing the mixed transfer until both approximation sites
        # in the pair are current.
        PeriodicProductsLab.invalidate!(ctx.mixed_cache, target_site)
    end
    return nothing
end

# Adapt scheduling positions to the reviewed cache API without coupling the two
# lower-level packages. The cache does not use the completed-sweep counter.
function _cache_state(st::LatticeIteratorLab.SweepState{D}, site::Int=st.site) where {D}
    direction = D === LatticeIteratorLab.Forward ? PeriodicProductsLab.Forward : PeriodicProductsLab.Backward
    return PeriodicProductsLab.SweepState{direction}(site, st.completed_sweeps)
end
