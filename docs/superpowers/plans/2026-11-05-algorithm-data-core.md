# AlgorithmDataLab synchronous core implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement task by task.

**Goal:** Implement the approved synchronous scientific-data stage with small
English REPL examples, independently of execution and logging packages.

**Architecture:** A local DataStore publishes a versioned JSON index via a
same-directory atomic rename. Immutable typed references identify inputs,
executions, snapshots, and measurement attempts within one store. Tensor copies
use Julia Serialization in separate payload files; result files use plain JSON.
Each mutation uses the store's lock; there is one externally coordinated writer
per directory. Reads inspect the committed index and load payloads only on request.

**Tech Stack:** Julia 1.12, AlgorithmConfigLab, JSON 1, Serialization, UUIDs, Dates.
**Spec:** ../specs/2026-11-05-algorithm-data-and-log-design.md

## Global constraints

- No ExecutionLab or AlgorithmLogLab dependency, internal tasks or thread creation.
- Copy mutable state before snapshot persistence; default copy_state is deepcopy,
  and custom scientific storage may require a specialized adapter.
- All measurement attempts/results are retained. Snapshot retention is independent;
  a persistent pin prevents deletion of a payload required by a checkpoint.
- Configurations are explicit structs; reuse AlgorithmConfigLab JSON conversion.
- Config/result metadata is frontend-readable; no evaluation of stored Julia text.
- Large tensor payloads require trusted Julia reconstruction, not frontend access.
- English examples are top-level, independently runnable, and inspectable in REPL.
- New sibling package ends in Lab. Do not bump existing versions or commit work.

## Review focus

1. Out-of-store references must fail before modifying metadata or copying payloads.
2. Encoding/write failure must not advertise a completed result or snapshot.
3. Pruning preserves checkpoint pins but not historical measurement input references.
4. Nested mutable storage must detach; repeated persistence of one capture writes once.
5. Reopened readers see only committed metadata and distinguish missing/removed inputs.

## Implementation choices

- DataStore(path) creates only in an empty directory or opens an existing store.
- Capture IDs also identify payloads; a repeated save of the same capture is
  idempotent while retained. Independently captured states are not deduplicated.
- Retention uses explicit delete_snapshot! and persistent pin_snapshot!/release_pin!
  primitives; callers decide which steps/final state to capture and retain.
- Checkpoint files/restoration stay external. Pin first, publish a checkpoint
  referring to that snapshot, and release its pin only after removing the checkpoint.
- Julia Serialization is the initial binary format: use trusted files and the
  recorded Julia/package environment. No cross-version binary guarantee.
- result_data(result) defaults to named fields of an AbstractMeasurementResult;
  specialized adapters convert opaque/complex results to explicit plain data.
- Numerical measurement exceptions produce failed attempts. Encoding and storage
  errors propagate; incomplete attempts remain visibly running, never completed.
- Single index publication is atomic local-filesystem visibility, not fsync durability
  or multi-process transactions. Failure may leave unreferenced payload files; readers
  ignore them. No automatic destructive orphan cleanup in this stage.
- Native implementation is authorized by the user's request to start AlgorithmData;
  no additional planning approval needed. One independent review at the end.

## Task 1: Store, identities, and provenance

Files: Project.toml, src/AlgorithmDataLab.jl, src/types.jl, src/storage.jl,
src/encoding.jl, src/records.jl; test/store.jl (including provenance tests).

Interfaces: DataStore(path), record_input!(store; execution=nothing, step=nothing,
label="", metadata=(;)), qualified record_execution!(store, config; parent=nothing,
trigger_step=nothing, constructors=(), provenance=(;)), finish_execution!,
info(store, ref), records(store, kind).

- [x] Write tests: create/reopen, invalid marker, nonempty directory, external input,
  nested execution ancestry vs input provenance, cross-store rejection, invalid
  metadata rejection without a partial record, and exact JSON index values.
- [x] Run RED, implement synchronous atomic storage and plain-value validation,
  config encoding through writeconfig, then run tests GREEN.

## Task 2: Snapshot ownership and retention

Files: src/snapshots.jl; test/snapshots.jl.
Interfaces: capture(input_ref, value), save_snapshot!(store, captured; metadata),
load_snapshot(store, snapshot_ref), qualified copy_state, delete_snapshot!,
pin_snapshot!(store, snapshot_ref; owner), release_pin!(store, pin_ref).

- [x] Tests: nested copy isolation, repeated save identity, reopen and lazy loading,
  never-persisted versus removed inputs, two pins sharing one payload, deletion
  ordering, latest-only retention by explicit pruning, serialization failure and
  failed index publication not publishing a completed snapshot.
- [x] Run RED, implement payload write before index commit and reference-aware
  deletion (commit removal before deleting bytes), run all current tests GREEN.

## Task 3: Measurement records

Files: src/measurement.jl; test/measurements.jl.
Interfaces: AbstractMeasurement <: AbstractAlgorithmConfig,
AbstractMeasurementResult, measure, measure!(store, definition, input; source,...),
measure!(store, definition, snapshot_ref), load_result, qualified result_data,
start_measurement!, finish_measurement!, fail_measurement!, load_measurement.

- [x] Tests: synchronous no-copy input, independent attempt IDs and configurations,
  actual result arrays, later measurement on reopened snapshot, computation failure
  retained and retry distinct, unsupported result encoding propagated, invalid status
  transitions, source and execution ancestry independent, no completion before payload.
- [x] Run RED, implement methods and extend current suite to GREEN.

## Task 4: Presentation, examples, shared measurement identity

Files: src/show.jl, README.md, examples/01_snapshots.jl,
02_measurements.jl, 03_retention_and_checkpoints.jl, 04_provenance.jl,
05_measurement_failures.jl, examples/README.md; test/show.jl, test/runtests.jl.
Minimal integration: LoopTNRLab/Project.toml and src/measurement/interface.jl
import the same measurement hierarchy from AlgorithmDataLab. No scientific algorithm
or execution behavior changes; broader integration remains stage 4 of the spec.

- [x] Add concise show/qualified describe, document file format, ownership, failure
  semantics, binary trust/environment limits and caller-controlled retention.
- [x] Write top-level English examples with values visible after each numbered step.
- [x] Verify all new-package tests and examples with one BLAS/Julia thread. Check
  shared measurement binding in isolation; no broad LoopTNR numerical test run.
- [x] Independent read-only final review, fix substantive findings with RED/GREEN
  tests, report evidence and remaining scope. Do not commit or bump versions.

## Execution ledger

- Preflight: AlgorithmDataLab does not exist; AlgorithmLogLab is committed.
  MPSSplitLab was only inspected and remains unchanged. Stage 2 is now explicitly
  requested. No background-execution or scientific-integration expansion.

- Tasks 1–3: test-first implementation. Missing store/snapshot/measurement APIs
  were observed RED before implementation; cumulative GREEN counts were 28, 59,
  and 99. Snapshot tests include actual serialization and index-publication failure,
  not mocked success paths. Measurement tests distinguish calculation failure from
  result encoding/publication failure.
- Task 4: show/describe tests first failed on the absent describe method, then 109
  assertions passed. Strict schema and duplicate normalized dictionary-key tests
  exposed three failures; fixed at the JSON boundary, then 112 assertions passed.
- Shared interface: three isolated identity assertions initially failed against
  LoopTNRLab's independent definitions, then passed after replacing them with imports.
  LoopTNRLab changes are Project.toml, its resolved Manifest.toml, and the two-line
  measurement interface. Dependency resolution preserved existing versions and
  disabled automatic precompilation; no scientific algorithms or numerical tests
  were run. The three shared identity assertions passed again after final fixes.
- Independent final read-only review found one substantive issue: repeated snapshot
  persistence returned inside the transaction callback, so the index still rewrote.
  A filesystem identity/mtime regression failed before the fix. Snapshot publication
  now returns under the lock before any write for retained captures. Final suite:
  114 assertions passed with one Julia/BLAS thread. Five English top-level examples
  cover snapshots, measurements, pins/retention, provenance, and failure/retry.
- Scope retained: one coordinated writer; synchronous local storage; no background
  tasks, tensor-specific copy adapters, checkpoint restoration, fsync durability,
  or cross-version Julia binary guarantee. No existing version bump or commit.
