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

## Adopted: standard logging composition

AlgorithmLogLab now contains event schemas, field codecs and registry inspection.
Applications use `event_record(registry, event)` to obtain plain `event`/`data`
keywords, then use Julia Logging, LoggingExtras and LoggingFormats directly.
The custom logger, sink, file wrapper and Julia-internal exception routing are
removed. JSON uses LoggingFormats' standard layout, with recursive values and
optionally flattened keyword fields. Logging libraries belong to example/test
and application environments; the schema core has no hard dependencies.

Accept standard filtering and exception behavior. Validate outside logging macros
when errors must propagate. The package no longer promises damaged-tail checks,
failed-sink state, generated record UUIDs or custom logger descriptions.

## Evaluated: direct MLflow tracking

The user accepts upstream semantics and does not require the existing Ref tree,
event schema, or configuration interfaces to survive. A direct prototype is in
[evaluations/mlflow](evaluations/mlflow/README.md), using MLFlowClient 0.10.0,
MLflow 3.14.0, JLD2, and standard Julia logging composition. It has no Lab-package
dependencies or compatibility adapter. Thirty checks passed, including fresh
process reads after server restart, a TensorKit round trip, per-step metrics,
child measurements, memory-only asynchronous measurement, retained failure logs,
and HTTP artifact upload/download.

Feasibility was confirmed, but adoption is now deferred: the user prefers a
native Julia workflow without a Python tracking server. The prototype uses a
local/shared artifact directory for direct
large-tensor writes; the current Julia client's ordinary proxy API buffers full
files. Explicit Unix-millisecond timestamps avoid a confirmed client-default bug.
MLflow does not supply graphical configuration construction, typed Julia object
serialization, task scheduling, or our previous pin/provenance invariants.
JLD2 and ordinary Julia/application code supply the needed remaining behavior.
No production-package migration or deletion has occurred in this evaluation.

## Current candidate: DrWatson and a local file convention

The user requested DrWatson evaluation because MLflow's server/client setup adds
unwanted complexity. The standalone prototype is in
[evaluations/drwatson](evaluations/drwatson/README.md). It uses DrWatson 2.19.1,
JLD2, existing Julia logging packages, and DataFrames for result collection.
No server, Python runtime, Lab dependency, or compatibility adapter is involved.
The minimal example ran successfully, and 18 checks passed in a fresh Julia
process, including configuration/tensor/result restoration, log fields, shared
snapshot references, summary collection, and cache reuse semantics.

DrWatson supplies naming, saving/loading, Git tagging, optional caching, parameter
combinations, and summary collection. It does not supply run objects or enforce
run/state/measurement relationships. The prototype expresses those with a fresh
directory per run, relative snapshot paths, and small records. Large tensors stay
outside the directory scanned by `collect_results`.

`produce_or_load` is optional caching by filename, not an independent execution
record. It can reuse a file even when the calculation implementation changes.
Do not make it the default for repeated experiments. No package migration or
deletion has been performed during this evaluation.

## Adopted: Configurations.jl

AlgorithmConfigLab now uses Configurations 0.17.6 directly for `@option`, defaults,
and dictionary conversion. The custom constructor registry and JSON schema were
removed. Graphical `@config` input/display remains in AlgorithmConfigLab, while
Accessors still owns updates. JSON reads now take a concrete configuration type:
`readconfig(path, typeof(settings))`. No old-format compatibility is provided.

Configuration source and extensions decreased from 972 to 576 lines. The six
scientific/update packages now define options directly and no longer maintain
duplicate registration/default tables. Storage examples use plain configuration
metadata while native Julia restoration remains unchanged.

## Deferred candidates

These are recorded for later evaluation, not approved implementation work.
We may change our own APIs and goals to fit maintained libraries; exact behavior
compatibility and production migration are not requirements.

| Candidate | Possible responsibility | Boundary to evaluate |
| --- | --- | --- |
| [AbstractTrees](https://github.com/JuliaCollections/AbstractTrees.jl) | Configuration and data-reference tree traversal/display | It does not parse graphical constructor input; keep domain references separate from display views |
| [JLD2](https://github.com/JuliaIO/JLD2.jl) | Tensor/object storage backend | Small dense TensorKit round trips passed in the evaluation; broader tensor types and retention policies remain separate questions |

ExecutionLab already builds on Julia tasks and synchronization. Reassess whether
its cooperative step/pause protocol is needed before adopting a larger scheduler.

## Evaluation principle

Prefer deleting our generic machinery over wrapping third-party APIs to preserve
it. Start with a small real-use example, measure the resulting code and public
surface, and retain only behavior that serves an actual algorithm use case.
