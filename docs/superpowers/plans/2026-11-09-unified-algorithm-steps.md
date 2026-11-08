# Unified Algorithm State and Steps Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give ExperimentLab, PMPSLab and LoopTNRLab one state-to-generator interface.

**Architecture:** ExperimentLab owns `AbstractAlgorithmState` and the generic `algorithm_steps`. Algorithm packages extend the generic and keep their numerical state, event types and restoration logic. Session still accepts ordinary ResumableFunctions generators.

**Tech Stack:** Julia 1.12, ResumableFunctions, existing ExperimentLab storage and Session.

**Spec:** The approved conversation design: `AbstractSplitState <: AbstractAlgorithmState`, `LoopTNRState <: AbstractAlgorithmState`, and `start_run(algorithm_steps(state))` for both packages.

## Constraints and decisions

- Do not bump package versions or add dependencies.
- Remove the old `split_steps` / `tnr_steps` entry points, without compatibility wrappers.
- Export the shared interface from ExperimentLab; algorithm packages import it to define unqualified `@resumable` methods.
- Keep initialization, snapshots, restoration, progress types, return values and control semantics unchanged.
- Preserve arbitrary generator execution and arbitrary-value capture.
- Work on coordinated local branches in the three clean sibling checkouts, preserving their relative dependency paths. Do not publish or create PRs.
- Leave historical migration/design narratives intact; update active docs and examples.

## Review focus

- Nested LoopTNR delegation must still yield every internal split event.
- Pausing, resuming and cancellation must work through the shared generic.
- Disk snapshots and fresh-process restoration must resume without replay.
- A non-ALS split must extend the ExperimentLab generic and work inside LoopTNR.
- Existing plain generators and tensor snapshots must remain supported.

## Tasks

- [x] Add and run a failing ExperimentLab protocol test: a custom state implements the shared generic, pauses after a yielded update, resumes, returns its final result and supports default capture.
- [x] Add the ExperimentLab interface and its documentation; update the exported API test.
- [x] Migrate PMPSLab and LoopTNRLab implementations, callers, tests and active documentation. Add type/ownership assertions to existing interface tests.
- [x] Run ExperimentLab's tests, PMPSLab's optimization/execution tests, and LoopTNRLab's full suite including fresh-process restart.
- [x] Review the three diffs and remaining old-name references; record verification results.

## Execution record

User approved implementation of the interface described in the conversation. No additional design approval is needed. All three working trees were clean at the start.

- Protocol test initially failed on the missing abstract type and generic. After implementation, 28 assertions pass across cooperative/threaded control, continuation and default copying.
- ExperimentLab full `Pkg.test` passed with two Julia threads. PMPSLab interface test passed (14 assertions). PMPSLab and LoopTNRLab full `Pkg.test` suites passed with one Julia thread, including nested delegation, pause/resume, snapshots at every boundary and fresh-process restart.
- Fresh independent review found no actionable issues in generic ownership, state inheritance, nested delegation, snapshot semantics or active documentation.
- Final whitespace checks passed in all three repositories. Package versions, dependency declarations and manifests are unchanged.
- After verification, the user explicitly requested committing the changes, merging all three repositories into `main`, and pushing directly. This supersedes the initial local-only constraint; no PR is requested.
