# AlgorithmLogLab Core Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Native execution is authorized by the user's request to start; no further handoff approval is needed.

**Goal:** Deliver the first independently reviewable package: strict structured logs through Julia Logging, with English examples.

**Architecture:** AlgorithmLogLab is a sibling leaf package. Registered immutable event structs define fields; explicit codecs handle special values. One logger routes opted-in modules to durable JSON Lines and optional terminal output, forwarding other modules to their original logger.

**Tech Stack:** Julia 1.12, Logging, Dates, UUIDs, JSON 1, Test.

**Spec:** ../specs/2026-11-05-algorithm-data-and-log-design.md

## Global Constraints

- No ExecutionLab, AlgorithmDataLab, AlgorithmConfigLab, TensorKit, or MPS dependency.
- No existing package version changes; the new package starts at 0.1.0.
- No scientific-package migration, numerical tests, frontend, or scheduler in this stage.
- Strict encoding, no arbitrary struct reflection except registered events, no string fallback.
- Info/Warn/Error durable by default; terminal filtering independent; third-party modules bypass strict validation.
- Append-only output; exceptions propagate; a failed sink cannot append after a partial record.
- Work in new sibling /Users/chenqimeng/Documents/GitHub/AlgorithmLogLab on codex/algorithm-log-core.
- Leave changes uncommitted for the user's staged review/commit workflow.

## Review Focus

- Unsupported or non-finite values must not yield plausible but altered JSON; Task 1.
- Module routing must not let missing events bypass validation or swallow third-party records; Task 2.
- Terminal filters and framework exception handling must not disable disk records or hide failures; Task 2.
- Concurrent task writes and scoped execution IDs must not interleave records or overwrite context; Task 3.
- Reopening an incomplete log must reject appending rather than silently corrupt history; Task 3.

## Files and interfaces

Create in the sibling package:

- Project.toml, AGENTS.md, .gitignore: standalone package and version constraints.
- src/AlgorithmLogLab.jl: small entry module with four exports.
- src/events.jl: AbstractLogEvent, EventRegistry, strict field encoding.
- src/logger.jl: AlgorithmLogger and standard Logging routing.
- src/files.jl: with_logfile, append/reopen lifecycle.
- test/runtests.jl plus focused event/logger/file test files.
- examples/01_structured_events.jl, 02_validation_and_codecs.jl,
  03_routing_and_context.jl, examples/README.md, README.md.

Exports: AbstractLogEvent, EventRegistry, AlgorithmLogger, with_logfile.
Qualified public author helper: event_record(registry, event), returning encoded
event identity and data for direct schema inspection. Concrete immutable event
types only in this first version; parametric types must be concretely registered.

## Task 1: Registered event schemas and encoding

Produces EventRegistry(pairs...; codecs=()), where each pair maps a unique stable
String to a concrete immutable AbstractLogEvent subtype. Codecs explicitly map
concrete special field types to encoders producing supported plain values.
event_record(registry, event) returns event name and detached JSON-compatible data.

- [x] Write tests with literal expected data, duplicate IDs/types, mutable/unregistered
  events, unsupported nested objects, explicit codecs, non-finite floats, and exact
  integer preservation (reject integers outside the interoperable +/- (2^53-1) range).
- [x] Run tests and observe missing-interface failures.
- [x] Implement registration and strict encoding. Basic values: nothing, Bool,
  bounded machine integers, finite Float16/32/64, strings, tuples, named tuples,
  and vectors recursively containing these. Reject arbitrary dictionaries/structs,
  multidimensional arrays, and unsupported numeric types unless explicitly adapted.
- [x] Run focused tests to green.

## Task 2: Standard Logging integration

Produces AlgorithmLogger(io, registry; modules, execution_id=nothing,
min_level=Logging.Info, terminal=ConsoleLogger(stderr), fallback=current_logger()).
Exact explicitly opted-in modules only; no automatic child-module opt-in.
Only the `event` structured keyword is accepted on opted-in records.

- [x] Write failing tests for valid @info, missing/wrong event, unchecked extra
  keywords, third-party passthrough, independent terminal threshold, Debug opt-in,
  context encoding, and exceptions reaching the caller.
- [x] Implement logger methods; build full JSON text before writes, include format
  version 1, unique record ID, UTC timestamp, severity, execution ID, source
  metadata, message, stable event identity, and typed data.
- [x] Test supported values through actual JSON parsing rather than only inspecting
  intermediate structures. Forward unmanaged messages with their original metadata.
- [x] Run all package tests to green.

## Task 3: Append lifecycle and shared task contexts

Produces with_logfile(f, path, registry; kwargs...), opening append mode and
executing f(logger) under with_logger. AlgorithmLogger(parent; execution_id)
creates a context view sharing registry, sink lock, failure state and output policy.

- [x] Write failing tests for append/reopen, close on exceptions, incomplete trailing
  record rejection, actual closed-stream I/O failure, a failure-latched sink, concurrent
  tasks emitting intact JSON Lines, and child-context isolation.
- [x] Implement synchronous writes/flush with one shared lock and a poisoned-sink
  guard after I/O failure. Do not create background tasks in the package.
- [x] Run all package tests; run examples in fresh small Julia processes or one
  driver with isolated modules. No broad ecosystem regressions.

## Task 4: Reviewable examples and final verification

- [x] Write English README and three feature-focused examples; each uses real
  Logging calls and assertions on emitted records. Explain strict module scope,
  record shape, schema naming, special codecs, JSON Lines publication, ownership,
  terminal/fallback semantics, and that close/flush is not a power-loss guarantee.
- [x] Run package test entry point and examples with one Julia/BLAS thread;
  use --compile=min -O0 to keep resource usage low.
- [x] Fresh read-only code review; repair substantive findings with regression tests.
- [x] Report size, validation and the first example to review. Do not claim the
  later AlgorithmDataLab/execution stages have been implemented.

## Execution record

- Tasks 1–4 complete in the new sibling repository on `codex/algorithm-log-core`.
  Changes remain uncommitted for review; no existing package version changed.
- Task 1 RED: EventRegistry absent; GREEN: 30 assertions passed.
- Task 2 RED: AlgorithmLogger absent; GREEN: 58 assertions passed.
- Task 3 RED: file/context entry points absent and failed sinks still appended;
  GREEN: 87 assertions passed. Closed IOStream writes were confirmed to return
  zero bytes; complete-byte-count checks now reject them.
- Additional foreign-expression policy regression: RED under blanket exception
  propagation; GREEN with scoped handling of Julia's synthetic logging errors.
- Fresh read-only reviewer found two P2 routing defects: JULIA_DEBUG overrides
  were blocked, and synthetic diagnostics were filtered a second time. Both
  reproduced in direct-versus-wrapped tests. The same fix pass also covered
  stateful destination filters being called more than once. All are green.
- Final verification: `Pkg.test(; allow_reresolve=false, julia_args=["--compile=min",
  "-O0", "--threads=2"])` passed 100 assertions. All three examples ran afterward.
  Evidence: `/tmp/algorithmlog-final.log`. No numerical ecosystem tests ran.
- Source size: 237 lines including docstrings/comments, across four files.
- Implementation choice: preserve standard routing using Julia 1.12's environment
  level helper and synthetic error report. Integration tests pin that behavior;
  future Julia changes may require an adapter update. This is documented in README.
- Review exclusions remain intentional: later data/execution stages, multiple
  independent writers to one path, and power-loss durability are not implemented.
  No substantive review finding remains unresolved.

## Follow-up: preserve types through event encoding

- User approved removing package-introduced `Any` from event encoding without
  changing constructor syntax, exports, JSON format, or package version.
- Added failing inference/schema tests first: original implementation produced
  6 failures and 5 errors, including heterogeneous dictionaries and accepted
  erased field declarations. Evidence: `/tmp/algorithmlog-types-red.log`.
- Codecs now use `FieldCodec{T,F}` entries in a parameterized tuple registry;
  event data and nested named tuples remain typed, and tuples retain their shape.
  Logger fields preserve the concrete registry and module-tuple types.
- Registration rejects abstract/Any schemas, including nested plain containers;
  explicit optional unions remain valid. User codecs remain responsible for
  their own inference, and opaque custom structs are not traversed.
- Updated README and example 02. Full scoped `Pkg.test` passed 129 assertions
  with two threads, and all three examples passed. Evidence:
  `/tmp/algorithmlog-types-final.log`. No numerical ecosystem tests ran.
- Same-script warmed allocation probes, Julia 1.12.5 at `-O1`: a small mixed
  record's field encoding fell from 2176 to 96 bytes; an adapted record from
  1472 to 64 bytes. Including JSON: 3152 to 1040 and 2560 to 992 bytes respectively.
  These are local illustrative measurements, not general performance guarantees
  or fixed allocation test thresholds. Evidence: `/tmp/algorithmlog-allocations-before.log`
  and `/tmp/algorithmlog-allocations-after.log`.
- Fresh static review identified a policy edge: fixed tuple fields containing
  explicit unions were rejected due to tuple covariance. A failing regression
  confirmed it; validation now checks those finite field alternatives directly.
  Open-ended tuple schemas remain rejected. No other actionable finding remained.

## Follow-up: simplify the logging boundary (supersedes the type restrictions above)

- User approved prioritizing reviewability over eliminating all dynamic dispatch.
  Keep NamedTuple event data and parameterized logger fields; restore encoder
  storage to `Dict{DataType,Any}` and remove tuple lookup machinery.
- Registration still requires a concrete immutable event type and unique names.
  It no longer restricts field declarations: actual values are checked during
  encoding, including nested values and codec output, before any bytes are written.
- Replaced inference assertions with encoding behavior checks in `test/encoding.jl`.
  Added broad-field, nested invalid-value, cyclic-container and no-partial-write
  coverage. The new broad-field acceptance test failed before the source change.
- Updated README and the interactive codec example to describe dynamic dispatch
  honestly. The four exports, user constructor syntax and JSON format are unchanged.
- Final scoped verification passed 141 assertions with two Julia threads, followed
  by all four REPL-oriented examples. Evidence:
  `/tmp/algorithmlog-simple-encoding-final.log`. No version bump or commit.

## Follow-up: human and machine inspection

- Implemented compact `show`, detailed `text/plain`, and qualified `describe`
  for AlgorithmLogLab loggers, registries, and internal sinks. No new exports.
- Descriptions use schema 1 with explicit object kinds, sorted complete registry
  mappings, and detached JSON-ready values. They summarize current settings/state;
  they are not event records or reconstructible checkpoints. Human registry
  rendering respects display limits. Codec captures, locks, IO contents and nested
  logger structures are not expanded.
- Sinks track the path known by `with_logfile`; caller-provided IO paths remain
  unknown, and custom IO open states remain unknown. Context views share this
  metadata and failure state. Display does not read/write/seek/flush the stream
  or invoke codecs/filters; destination minimum-level queries are documented.
- Added English REPL example 05 and documented the description schema. New tests
  failed first because `describe` was absent; final `Pkg.test` passed 194 assertions
  and all five examples passed. Evidence: `/tmp/algorithmlog-show-final.log`.
- Fresh independent static review found no actionable issues. No version bump,
  commit, or edits to other packages' displays were made in this stage.
