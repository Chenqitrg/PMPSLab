# Ecosystem reuse decisions

Updated: 2026-11-06. Scope: AlgorithmConfigLab, AlgorithmLogLab,
AlgorithmDataLab and ExecutionLab. Package versions must not change without
an explicit request.

## Adopted: Accessors and ConstructionBase

Use Accessors directly for configuration updates (`@set`, `@reset`, Julia dot
and index paths). Remove AlgorithmConfigLab's update macros and path engine.
Accept the libraries' ordinary semantics rather than preserve the old DSL.
Use standard constructors; put invariants in inner constructors. Do not add a
blanket configuration reconstruction adapter. ConstructionBase owner hooks
remain available for genuinely special types. Tree construction and JSON
interchange remain AlgorithmConfigLab responsibilities.

The old update code ignored construction metadata, dropping non-inferable type
tags and rejecting derived-field configurations. Removing that code shifts
reconstruction to the standard protocol. Tree/JSON keyword metadata no longer
claims to control updates; Julia properties and positional construction do.

## Deferred candidates

These are recorded for later evaluation, not approved implementation work.
We may change our own APIs and goals to fit maintained libraries; exact behavior
compatibility and production migration are not requirements.

| Candidate | Possible responsibility | Boundary to evaluate |
| --- | --- | --- |
| [LoggingExtras](https://github.com/JuliaLogging/LoggingExtras.jl) | Filtering, routing, file outputs, rotation | Prefer its standard logging semantics; keep only needed algorithm event conventions |
| [LoggingFormats](https://github.com/JuliaLogging/LoggingFormats.jl) | JSON logging | Check recursive structured values and string fallback against scientific/AI use |
| [AbstractTrees](https://github.com/JuliaCollections/AbstractTrees.jl) | Configuration and data-reference tree traversal/display | It does not parse graphical constructor input; keep domain references separate from display views |
| [JLD2](https://github.com/JuliaIO/JLD2.jl) | Tensor/object storage backend | Verify TensorKit round trips; provenance, retention and pin semantics remain domain concerns |
| [Configurations](https://github.com/Roger-luo/Configurations.jl) | Nested configurations, defaults, dictionary/TOML conversion | Evaluate its `@option` type-definition style and maintenance activity; graphical tree input is not built in |

ExecutionLab already builds on Julia tasks and synchronization. Reassess whether
its cooperative step/pause protocol is needed before adopting a larger scheduler.

## Evaluation principle

Prefer deleting our generic machinery over wrapping third-party APIs to preserve
it. Start with a small real-use example, measure the resulting code and public
surface, and retain only behavior that serves an actual algorithm use case.
