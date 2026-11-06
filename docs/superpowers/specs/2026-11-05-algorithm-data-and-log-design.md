# AlgorithmDataLab and AlgorithmLogLab

Status: approved for staged implementation. AlgorithmLogLab is the first stage;
the data-storage and external-execution interfaces below remain design proposals
until their own implementation stages. Illustrative API names are not guarantees.

This document records the design discussion of 2026-11-05. Sections marked
“proposed implementation choice” contain choices made while assembling the
draft, rather than additional requirements approved by the user.

## Purpose

Keep scientific states and results available for analysis, additional
measurements, and a future frontend. Preserve structured algorithm diagnostics
without coupling either responsibility to execution control.

The first scientific use case saves the tensor and CFT spectrum after each
outer algorithm step. MPS storage is off by default; when enabled, the usual
choice is the final optimized MPS, with local states optionally recordable.
These are application defaults, not concepts built into the generic packages.

All new package names end in `Lab`. Package versions remain unchanged unless
the user explicitly requests a patch bump. There is no production compatibility
requirement, but persisted format versions must be explicit.

## Package boundaries

| Owner | Responsibility |
| --- | --- |
| AlgorithmConfigLab | Configuration construction, display, and configuration codecs |
| AlgorithmDataLab | Scientific state snapshots, measurement interfaces and records, scientific result storage and retrieval, provenance |
| AlgorithmLogLab | Typed diagnostic events, strict validation, structured encoding, logging sinks and presentation |
| Scientific algorithm packages | Concrete measurement configurations and calculations, scientific result types, diagnostic event definitions, tensor-specific adapters |
| External execution/integration layer | Scheduling, task lifecycle, context propagation, concurrency, memory admission, waiting, retries, and failure policy |

AlgorithmDataLab has **no dependency on ExecutionLab**, no ExecutionLab
extension inside it, and no internal session or scheduling machinery. Its
operations are synchronous primitives that an external layer can schedule.
It can use AlgorithmConfigLab for measurement configurations.

AlgorithmLogLab does not read AlgorithmDataLab records and does not depend on
AlgorithmDataLab, ExecutionLab, or scientific packages. It uses Julia's standard
Logging interface. External composition supplies the execution context and
connects its sinks to the run directory.

Costs, residuals, timings, convergence diagnostics, warnings, and errors belong
to logs. Tensors and scientific measurements such as spectra belong to data.
The distinction is their purpose, not whether their representation is numeric.

Algorithm-essential stopping state remains owned by the algorithm. Changing
logging filters or retention must never change convergence decisions.

## Scientific data model

Use separate concepts for the following:

- **Execution record:** identity, configuration, parent execution and triggering
  step when applicable, reported status, and provenance. It records facts supplied
  by the external executor; it does not control execution.
- **Input/state identity:** the state a calculation used, even when no persistent
  tensor snapshot exists. A supplied external tensor does not require a fictitious
  iterative run or step zero.
- **Snapshot record:** source identity and step, descriptive metadata, and a
  reference to a tensor payload when it is retained.
- **Measurement definition:** an immutable, reconstructible configuration for
  a scientific calculation.
- **Measurement attempt:** a distinct execution of a definition on specified
  input, with identity, configuration, source references, status, and either
  results or structured failure information.
- **Scientific result:** a typed payload defined by the scientific package.

Nested executions form a parent/child tree. Each can have its own configuration,
steps, results, and logs. This does not require a directory per child execution.
Execution ancestry and input provenance are different relationships: a new
measurement can consume an old snapshot without pretending to be an execution
that happened during the original algorithm step.

A measurement is an algorithm, potentially an iterative one. It normally records
its final scientific result, but may expose intermediate steps and scientific
data using the same mechanisms. Its intermediate costs and residuals remain logs.

Re-running a measurement, including with different parameters or after failure,
creates a new attempt. It never overwrites an earlier result or failure record.

## Measurement interface

The agreed shape is an explicit configuration type and Julia method dispatch,
not an arbitrary persisted function or closure:

```julia
abstract type AbstractMeasurement <: AbstractAlgorithmConfig end
abstract type AbstractMeasurementResult end
function measure end
```

The result base type is retained from the existing LoopTNRLab interface as a
proposed implementation choice. Concrete type names below are illustrative:

```julia
struct SpectrumMeasurement <: AbstractMeasurement
    nlevels::Int
    tolerance::Float64
end

measurement = SpectrumMeasurement(20, 1e-10)
# The scientific package implements measure(::SpectrumMeasurement, tensor).
result = measure(measurement, tensor)
```

Concrete configuration types should also provide the keyword constructors
expected by AlgorithmConfigLab. Public API names and exports will be minimized
during implementation planning; these snippets are not runnable examples yet.

Definitions contain parameters, not tensors, tasks, paths, or runtime handles.
Their stable registered identifiers and encoded parameters allow reconstruction
through explicit constructor mappings. Reuse the existing configuration codec
mechanism where applicable; do not assume ordinary file I/O automatically
discovers every `config_codecs` method.

Reconstruction requires the corresponding implementation package. Record its
version and available source identity; saving configuration does not save source
code or guarantee bitwise reproducibility.

Each measurement is independent in the first version. There is no dependency
graph, automatic common-subexpression reuse, or measurement-result cache.
Repeated computation is acceptable. A scientific package may organize work
internally within one measurement.

Measurements treat inputs as read-only. A method that needs destructive work
must make its own workspace. Mutable results must not remain aliased to a live
algorithm buffer when queued for asynchronous persistence.

## State ownership and measurement modes

| Mode | Input | Required isolation |
| --- | --- | --- |
| Synchronous measurement | Current in-memory tensor | No snapshot copy needed while the algorithm is not modifying the input |
| Background measurement | Independent in-memory snapshot | Copy before the algorithm resumes mutation |
| Later measurement | Tensor loaded from a saved snapshot | No second copy required if the measurement is read-only |

Saving a tensor snapshot always requires an independent copy. There is no
ownership-transfer or unsafe zero-copy snapshot option. The copy must detach
actual mutable tensor storage, not merely an outer container; scientific adapters
must establish this for their supported types.

Multiple read-only consumers may share that independent copy. A snapshot and a
checkpoint for exactly the same state share the copy and stored tensor payload.
No content-hash deduplication across independently created snapshots is required.

When tensor persistence is disabled, a background measurement uses only a
temporary in-memory snapshot. Release it after its last consumer finishes. Do
not create a temporary tensor file by default. Remote backends may eventually
need a different transport mechanism; that is outside the initial backend.

Saving state and doing a measurement are independent choices. Synchronous
measurement can persist a spectrum without ever saving its input tensor. Such
a record has input provenance but cannot promise that its input can be reloaded.

## Retention and checkpoints

All computed scientific measurement results are retained, including repeated
attempts. There is no automatic latest-only policy for small measurement results.
Measurement timing can still be configured to control computation cost.

Large tensor snapshots have their own capture and retention policies: all steps,
selected steps, latest/final state, or disabled. Concrete policy names are left
to the builder. MPS follows its own application-level capture policy.

A checkpoint is a restart description, not another mandatory tensor copy. It
references the shared tensor payload and adds whatever state the algorithm needs
to resume. Algorithms decide which caches are rebuildable and which state must
be saved. AlgorithmDataLab does not implement algorithm restoration.

Deletion respects all live persistent references, including checkpoints. Keeping
only the latest analysis snapshot does not permit deleting a tensor still needed
by a retained checkpoint. A measurement's historical source identity is not an
implicit permanent pin: otherwise retaining all measurement results would force
retaining all tensors. Metadata must distinguish retained, deliberately removed,
and never-persisted inputs. Active consumers must finish before their transient
storage is released.

The precise reference-management protocol is an implementation choice, but these
deletion and sharing invariants are mandatory. Checkpoint construction/restoration
remains external; no new universal checkpoint engine is part of this project.

## Execution contract

Background measurement and persistence are allowed, but scheduled externally.
Synchronous use remains fully supported. The default choice of local execution
backend is left to the integration layer, separate from the default waiting rule.

The external coordinator must:

1. Reserve capacity before allocating a snapshot copy.
2. Copy while the producer is not mutating the source.
3. Schedule consumers and allow the algorithm to continue.
4. Account for snapshots until the last consumer releases them.
5. Wait for capacity when the configured limit is reached; never silently drop
   a record to keep computation moving.

The snapshot budget bounds queued/in-use snapshot storage, not all memory used
by the numerical algorithm or measurement workspaces. As a proposed implementation
choice, a snapshot larger than the entire configured budget produces an explicit
configuration error instead of deadlocking or exceeding the budget silently.

By default a run returns only after computation, measurements, and persistence
finish. An external asynchronous entry point can return a handle earlier. It
must distinguish main computation finished from all submitted work settled.

Measurement calculation failures are recorded and do not stop the main algorithm
or unrelated measurements. They remain visible in the final outcome; completion
with measurement failures is not reported as complete success. No automatic retry
is required in the first version. Re-measurement creates a new attempt.

Persistence failures stop further algorithm advancement when detected. Work
already in flight may complete. Report the error and retain unsaved objects when
memory permits; do not claim uncommitted data is durable. Validation/encoding
errors are programming or persistence errors, not ordinary failed measurements
to silently skip. This classification is a proposed implementation choice.

## Storage and frontend access

Each run/data directory is self-contained. There are no references to tensor
payloads in another run directory. Standalone input states and subsequent
measurements may live in a directory without an iterative parent run.

Conceptual layout (exact filenames are an implementation choice):

```text
run/
  metadata/
  tensors/
  snapshots/
  measurements/
  checkpoints/
  logs/
```

Metadata, configuration, status, errors, and common scientific results must be
readable without Julia, using JSON or explicitly documented interoperable
encodings. Special result types require explicit encoders. The stored format has
its own version, independent of package versions. The reader must not evaluate
arbitrary Julia code from files.

Tensor payloads only need Julia reconstruction. Frontends do not display them;
they may show availability, origin, and storage metadata. Binary backend and
tensor codec choices are left to the builder. Numerical arrays inside measurement
results are still scientific result data, not automatically tensor snapshots.

Frontend readers may inspect a running directory. Publish completion only after
the referenced payload is completely written; incomplete files must not appear
as completed results. Pending/running/failed records can be visible without a
completed result reference. Arrival order is not algorithm-step order.

Proposed implementation choices:

- One externally coordinated publisher per directory in the first version;
  concurrent workers report results through it. Do not build distributed
  multi-writer transactions now.
- Use staged writes and an explicit publication boundary on a local filesystem.
  A reader can ignore a trailing incomplete log record; complete prior records
  remain readable. Exact atomic-write and recovery mechanisms belong in the plan.
- Define logical committed visibility first; stronger power-loss guarantees and
  distributed filesystem behavior are not implicitly promised.
- Readers load indexes/metadata without eagerly loading tensors. A future
  frontend and remote workers are consumers, not deliverables of this first version.

## Structured logging

Use standard Julia `Logging`, with explicitly registered event structs:

```julia
struct UpdateCompleted <: AbstractLogEvent
    step::Int
    cost::Float64
    duration::Float64
end

@info "Update completed" event=UpdateCompleted(12, 1e-8, 0.4)
```

The event's fields are its schema. One meaningful event contains the related
diagnostic values available at that time. A later warning or background completion
is a separate event, not an edit to an earlier one.

An outer envelope supplies common metadata such as execution identity, timestamp,
level, and registered event identity. A step field intrinsic to an event is not
required to be repeated in the envelope. Display messages are not machine-readable
event identifiers.

Event types register stable stored names and format/schema information. For
registered types, supported basic fields encode automatically. Special field
types require explicit codecs; unsupported fields fail. There is no arbitrary
object traversal, `show`/string fallback, or automatic tensor summary. Authors
must explicitly supply summary fields if that is what they want to record.

Non-finite numbers, complex values, and other non-native JSON values need a
documented explicit representation or rejection; never silently replace them
with null or strings. Exact supported basic types are a builder decision.

Strict validation applies to explicitly opted-in modules, not to whichever
messages happen to contain an `event` key. Missing events in an opted-in module
are errors. Structured diagnostic fields belong in the event, not arbitrary
unchecked extra key/value fields beside it. This is a proposed implementation
choice that preserves the accepted schema constraint.

Unadapted third-party messages retain ordinary logging behavior and are not mixed
into the structured algorithm log file. Adapters can opt them in later. Do not
force dependencies to implement this package's conventions.

Logs are append-only. A retry or later completion adds a new event; it does not
erase failures or rewrite past events. The external layer/frontend may summarize
current state without rewriting event history.

Disk defaults to Info/Warn/Error, with Debug opt-in. Terminal verbosity is
independent. Persisted cost/convergence events use Info rather than depending on
Debug being enabled. Terminal filtering must not suppress the disk sink.

Errors from strict validation and durable sinks must reach the external caller;
implementation must explicitly check Julia Logging's filtering and exception
handling behavior rather than assume a thrown logger error always propagates.

## Existing code and migration constraints

Read-only inspection found:

- `LoopTNRLab/src/measurement/interface.jl` already defines
  `AbstractMeasurement <: AbstractAlgorithmConfig`, `AbstractMeasurementResult`,
  and `measure`. Reuse/move the interface rather than create a parallel hierarchy.
- `LoopTNRLab/src/measurement/history.jl` provides an in-memory result vector.
  Persistence, attempt identity, and input provenance require additional design;
  this vector alone is not the new store.
- `MPSSplitLab/src/logs.jl` contains site/sweep diagnostics. They are candidates
  for typed log events, not scientific data records.
- `MPSSplitLab/src/split/state.jl` currently reads error history from its log to
  decide stopping. Separate algorithm-owned stopping state before making logs
  optional or externally streamed.

Existing CFT implementations, tensor codecs, checkpoint behavior, and execution
backends must be checked during implementation planning. This design does not
claim they already implement the proposed behavior.

## Scope and review sequence

Build small independently reviewable cores before integrating numerical packages:

1. AlgorithmLogLab event registration, strict encoding, standard Logging routing,
   append-only output, and English examples.
2. AlgorithmDataLab synchronous state/result storage, explicit measurements,
   provenance, retention, and English examples using small inputs.
3. External local execution composition demonstrating bounded snapshot ownership,
   background measurements/writes, waiting, and failure propagation.
4. Focused migration of existing scientific interfaces and logging, with numerical
   stopping state isolated from retained logs.

This sequence is proposed, not a request to start implementation. New packages
will be siblings under the GitHub parent directory. Do not scaffold them, modify
runtime code, commit implementation, or bump versions as part of reviewing this
document. No broad numerical regression runs are needed for the document.

Initial non-goals: a frontend, a cluster scheduler, a measurement dependency graph,
cross-run tensor deduplication, Python tensor reconstruction, automatic code
archiving, and a universal checkpoint/restart framework.

## Acceptance scenarios for the eventual implementation

- Synchronously measure an in-memory tensor without saving or copying that input;
  persist the small result and retain its provenance.
- Copy one state, resume producer mutation, and obtain correct independent
  background measurements and persistence from the copy.
- Share one payload between a snapshot and checkpoint; deleting one reference
  leaves the other's data usable.
- Retain every spectrum while pruning old unpinned tensor payloads; metadata
  truthfully reports that some old inputs can no longer be loaded.
- Re-measure a saved state with changed parameters; both attempts remain readable.
- Read completed results during writing without exposing incomplete payloads.
- Wait under snapshot memory pressure; stop advancement on write failure; continue
  after a recorded numerical measurement failure.
- Distinguish main calculation completion, all work settled, and all work succeeded.
- Preserve nested execution context while keeping measurement input provenance
  independent from execution ancestry.
- Encode a registered diagnostic event without a handwritten per-event JSON
  function, reject unsupported values, and leave unrelated third-party logs alone.
- Retain disk cost records while terminal output is suppressed, without affecting
  algorithm convergence.

Implementation tests should target these boundaries with small synthetic payloads
and deterministic failure cases. Numerical validation should be limited to the
packages and behavior actually changed, with controlled resource usage.
