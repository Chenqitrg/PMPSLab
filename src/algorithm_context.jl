# ------------------------------------------------------------------------------
# AlgorithmContext — mutable numerical working state
#
# A context owns the algorithm-specific tensors and derived environments needed
# by local numerical updates. For example, SplitContext contains the target and
# working MPS values, transfer matrices, and partial-product caches.
#
# Execution concerns such as the algorithm configuration, next safe step,
# convergence log, elapsed time, and completion state belong to an
# AbstractAlgorithmRun that contains the context. Concrete contexts expose only
# the numerical operations their algorithms require; no cache protocol is
# imposed until multiple implementations demonstrate a common interface.
# ------------------------------------------------------------------------------

abstract type AlgorithmContext{E, S} end
