# ------------------------------------------------------------------------------
# AlgorithmContext — generic protocol for loop MPS algorithms
#
# Every algorithm context holds a target object, a working object and a
# result log. Implementations (e.g. SplitContext) add algorithm-specific
# caches and transfer matrices.
#
# Protocol (must implement):
#   build_cache(ctx)       — initialise caches from target + working
#   rebuild_cache(ctx)     — full rebuild (e.g. after wrap-around)
#   advance(ctx)           — incremental cache slide after a site update
#   update(alg, ctx)       — one unit of work (step / site / sweep)
# ------------------------------------------------------------------------------

abstract type AlgorithmContext{E, S} end
