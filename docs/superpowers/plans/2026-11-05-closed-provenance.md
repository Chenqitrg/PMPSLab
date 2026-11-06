# Closed provenance implementation plan — reverted

> Use superpowers:executing-plans inline; no automatic commits.

**Goal:** Bind every captured state and measurement to a fixed-config execution,
without manual source references or implicit tensor persistence.
**Architecture:** An ExecutionRecord owns a store/identity handle. capture(run,
step, value) copies the value and atomically publishes input plus snapshot metadata.
Measurements and explicit persistence consume that captured state; result/config
binary loading and frontend JSON views remain separate.
**Tech Stack:** Existing Julia/JSON/Serialization dependencies only.
**Spec:** The user-approved design in this conversation, summarized below. It
supersedes optional external provenance in the original synchronous-core plan.

## Approved constraints

- Closed workflow: no standalone/external input registration or raw-input measure!.
- One fixed configuration per execution; changing parameters means a new execution.
- Copy tensor before asynchronous handoff. Default: no tensor file. Only explicit
  save_snapshot! writes tensor data. A shared read-only copy serves measurement/save.
- All records inherit execution and step; caller never supplies a source link.
- Configurations/results restore Julia objects; retain JSON for inspection.
- Data operations remain synchronous. Scheduling and checkpoint restoration stay outside.
- No package version changes, no commits, no unrelated numerical package tests.
- English top-level examples; short source files; avoid unnecessary exported helpers.

## Interfaces and decisions

- record_execution!(store, config; parent=nothing, trigger_step=nothing, label,
  constructors, provenance) -> ExecutionRecord(store, id). Config must be immutable
  at the top level and is deep-copied before binary/JSON storage. No update API.
- Qualified load_execution(store, execution_ref) reopens a handle;
  load_config(run) returns a detached configuration. finish_execution!(run; status)
  records producer completion; preexisting captures can still finish background work.
- capture(run, step, value; label, metadata) -> StateSnapshot(run, id, source, value).
  It copies storage and publishes input+snapshot metadata, never tensor bytes.
  Requires a running execution; step is a nonnegative exact JSON integer. Multiple
  states at a step can have distinct identities; numbering is supplied by algorithm.
- save_snapshot!(captured; metadata) -> SnapshotRef. Returns existing identity
  without writing on repeat. load_snapshot(store, ref) still returns raw Julia data.
- measure!(captured, definition; constructors, metadata) and
  measure!(store, definition, saved_ref; constructors, metadata) inherit provenance.
  The saved-ref form loads only a stored snapshot. No unrelated execution override.
- Qualified start_measurement! uses the same captured/saved forms. Coordinators
  finish/fail attempts with existing store/ref APIs. measure! uses a detached
  definition for both persistence and computation.
- Captures always publish a snapshot identity with stored=false until explicitly
  saved. Measurements point to it even when save happens later or never happens.
  Pruning never deletes metadata. No source record_input! API remains.
- info(store, run) and qualified describe(run) support inspection. Existing typed
  references and metadata listing remain the disk navigation primitives.

## Review focus

1. No input/snapshot/measurement can enter the supported API without an execution.
2. Mutable caller config or tensor data must not alter historical stored values.
3. Measuring before saving must keep exactly the same state identity after saving.
4. Delayed consumers may finish after producer completion; new captures may not.
5. Publication failures must leave no falsely completed result or saved tensor.

## Task 1: Closed workflow and regression migration

Files: src/types.jl, records.jl, snapshots.jl, measurement.jl, storage.jl,
AlgorithmDataLab.jl, show.jl; test/closed_workflow.jl and existing test files.

- [x] Add failing tests for inherited links, config detachment, delayed measurement,
  explicit-only persistence, fixed execution config, completion boundary, missing
  source APIs, cross-store checks, and reopened provenance/config reconstruction.
  Run targeted test with one Julia/BLAS thread; expect missing new methods/behavior.
- [x] Implement interfaces above. Migrate old tests to execution-bound captures,
  retaining retention/pins, idempotence, binary/JSON restore and failure coverage.
  Snapshot-write failures now leave legitimate metadata with stored=false.
- [x] Run entire test/runtests.jl with --compile=min -O0 --threads=1; expect all pass.

## Task 2: Reviewable examples and documentation

Files: README.md, examples/01 through 05, new 06_async_handoff.jl, examples/README.md.
- [x] Update examples to begin with a configuration/run and auto-associated capture;
  no manual input record. Example 04 traces ancestry; 06 demonstrates external @async
  with an explicit gate, producer mutation, and frozen consumer data (no sleeps).
- [x] Explain in-memory snapshot identity vs stored payload, fixed configs, immutable
  provenance, explicit persistence, and coordinating all consumers before release.
- [x] Run all examples sequentially and inspect output. Independent read-only final
  review, fix substantive issues test-first, then whole suite. No commit/version bump.

## Execution ledger

- Preflight: existing package is uncommitted/untracked with working tests; no git
  baseline commit exists. Plan execution scripts assuming HEAD do not apply; record
  direct test commands/results here. User already authorized implementation.
- Shared interface: record_execution! produces ExecutionRecord; capture consumes it;
  measurement/save consume StateSnapshot with inherited store. No per-task agent work.
- Scope: data records cannot prove caller-supplied tensor was generated by the stated
  algorithm; external algorithm integration must use the recorded fixed config.

- Task 1: complete. Targeted closed-workflow tests first failed on missing load_config
  and capture(run, step, value). Implemented execution handles/fixed binary configs,
  automatic state metadata and inherited measurement links; migrated all regression
  tests to new APIs. Full suite reached 178/178, then 185/185 with publication-failure
  coverage and fresh-process execution-config restoration. One test expectation was
  corrected: AlgorithmConfigLab JSON vectors use a tagged values object.
- Task 2: six top-level English examples passed sequentially. External @async
  example gates the consumer until after tensor mutation and producer completion;
  measurement sees original values and no tensor file until explicit persistence.
- Verification: OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 Julia --startup-file=no
  --compile=min -O0 --project=AlgorithmDataLab; include(test/runtests.jl) then all
  examples. Exit 0; /tmp/data-closed-final.log. No heavy numerical dependencies run.
- User steering: no compatibility requirements. No old aliases or migration layer
  retained. Supported constructors remain extensible for internal scientific types;
  closed workflow means mandatory provenance, not a hardcoded whitelist of packages.
- Final independent read-only review: no actionable defects. Reviewer inspected
  source/tests/examples without running Julia. No commit/package version change.
- Final scope decisions: actual algorithm integration and verifying a producer's
  adherence to its configuration remain caller responsibilities; this stage stores
  provenance. Checkpoint restoration and scheduling remain external. Forged handles,
  mutation of documented read-only captured values, concurrent independent writers,
  and power-loss durability remain outside the supported contract. No additional
  compatibility or runtime machinery was introduced for those cases.

- User-requested rollback: reverted the entire closed-provenance implementation,
  not merely ExecutionRecord's show. ExecutionRef, explicit store arguments and
  record_input! are restored. Removed execution context/config-loading additions,
  closed-workflow tests and example 06. Retained the previously approved native
  load_result/load_measurement behavior. Tree-of-direct-parent links remains a
  design discussion, not an implemented follow-on. Rollback suite: 145/145.
