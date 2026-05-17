# Omni Distributed Refactor Blueprint

## Goal

This document proposes a refactor plan for the current multi-node / headless / replica management path in `vllm-omni`, with the following goals:

- align the architecture more closely with upstream `vllm`
- decouple single-node and distributed execution paths
- reduce the spread of coordinator-related state across the codebase
- isolate launch / registration / handshake concerns from request orchestration
- improve readability, testability, and long-term maintainability

This blueprint intentionally does **not** address the `diffusion` special path yet. The focus is the LLM path first.

## Current Problems

The current implementation works, but several concerns are mixed together:

- topology planning
- head vs headless mode selection
- replica registration and auto-assignment
- handshake and bootstrap logic
- coordinator lifecycle and wire protocol details
- dynamic replica attach / detach
- request routing and affinity
- stage-local client management

These concerns are currently spread across:

- `vllm_omni/entrypoints/cli/serve.py`
- `vllm_omni/engine/async_omni_engine.py`
- `vllm_omni/engine/stage_engine_startup.py`
- `vllm_omni/engine/orchestrator.py`
- `vllm_omni/engine/stage_pool.py`

The main issue is not just code size. The bigger issue is that distributed topology concerns and request-flow concerns are interleaved. As a result:

- single-node behavior is harder to reason about because distributed conditionals are present in common paths
- coordinator-related addresses and lifecycle details leak into modules that should not know about them
- `Orchestrator` knows too much about remote replica bring-up
- `StagePool` mixes routing, membership, lifecycle, and metrics
- distributed changes are likely to regress local behavior

## Upstream vLLM Design Principles Worth Borrowing

Upstream `vllm` keeps the distributed complexity relatively contained by using a small number of architectural boundaries.

### 1. Normalize topology early

In upstream `vllm`, CLI arguments are folded into a stable `ParallelConfig` early. Later modules consume the normalized config instead of repeatedly inferring:

- local vs remote
- head vs headless
- DP mode
- rank ownership
- coordinator requirements

This reduces mode-specific branching in runtime code.

### 2. Choose implementation once at the boundary

Upstream `vllm` uses small factories and abstraction boundaries:

- `Executor.get_class(...)`
- `EngineCoreClient.make_*()`
- `launch_core_engines(...)`

The key idea is: select the execution strategy once, then let the rest of the system talk to interfaces.

### 3. Concentrate launch / handshake logic

Upstream `vllm` places process bring-up and handshake logic in dedicated engine utility code, rather than spreading it through orchestration logic. This keeps:

- process lifecycle
- ZMQ address wiring
- coordinator address injection
- remote engine connection

away from request scheduling code.

### 4. Keep the coordinator as a control-plane component

In upstream `vllm`, `DPCoordinator` does not orchestrate requests directly. It is a control-plane component used for:

- publishing queue/load information
- wave coordination for MoE DP

It does not become a general-purpose dependency across unrelated layers.

### 5. Keep orchestration logic focused on request flow

The higher-level engine and request-processing layers mostly operate on abstract clients and do not directly own the distributed bootstrap path.

## Refactor Targets for vLLM-Omni

The target architecture should separate the system into the following layers.

### Layer A: Deployment planning

Introduce a pure-data planning object, for example `OmniDeploymentPlan`.

It should describe:

- runtime mode:
  - `single_node`
  - `distributed_head`
  - `distributed_headless`
- logical stages
- replica ownership
- local replica plans
- remote replica plans
- coordinator requirements
- membership-control requirements

This layer should not perform bootstrap or orchestration. It should only answer:

"What topology are we trying to run?"

### Layer B: Bootstrap / launch / handshake

Introduce a dedicated runtime/bootstrap layer responsible for:

- local replica spawn
- remote replica registration
- handshake
- address resolution
- replica readiness
- distributed-only attach hooks

This is the equivalent of the role played by upstream `vllm.v1.engine.utils`.

This layer should be the only place that knows details like:

- which side binds which socket
- how replica IDs are assigned
- how registration responses are encoded
- how headless replicas connect back to the head

### Layer C: Control plane

Keep these roles explicit and narrow:

- `OmniMasterServer`
- `OmniCoordinatorRuntime`
- `OmniCoordClientForStage`
- `OmniCoordClientForHub`

Their responsibilities should be:

- registration / route allocation
- heartbeat / liveness
- active replica snapshots
- control-plane publication only

They should not own request-routing semantics.

### Layer D: Membership and routing

Split the current `StagePool` responsibilities into smaller pieces:

- `ReplicaRegistry`
- `StageRouter`
- `StagePool`

Suggested responsibilities:

- `ReplicaRegistry`
  - live replica set
  - address-to-client mapping
  - attach / detach
  - stable replica identity

- `StageRouter`
  - load balancing policy
  - affinity bookkeeping
  - selection of serviceable replicas

- `StagePool`
  - stage-local submit / update / poll / abort facade
  - optional metrics accumulation

This separation removes distributed membership management from `StagePool`.

### Layer E: Request orchestration

`Orchestrator` should focus only on:

- request lifecycle
- stage transitions
- stage output handling
- abort/error cleanup

It should not directly own:

- remote replica construction details
- handshake logic
- coordinator wire protocol details
- registration payload interpretation

If dynamic attach/detach is needed, introduce a dedicated `MembershipController` that updates the `ReplicaRegistry`.

## Single-Node and Distributed Must Be Explicitly Separated

One of the main goals of the refactor should be to make single-node and distributed modes diverge early, instead of sharing a heavily branched runtime path.

### Single-node mode

In single-node mode:

- do not start `OmniMasterServer`
- do not start `OmniCoordinatorRuntime`
- do not construct hub/coordinator clients
- do not run registration / heartbeat / attach / detach paths
- use a static replica registry

### Distributed head mode

In distributed head mode:

- start `OmniMasterServer`
- start `OmniCoordinatorRuntime` if required
- create `ReplicaRegistry`
- create distributed membership controller
- allow remote replica register / attach / detach
- route through a dynamic replica snapshot source

### Distributed headless mode

In distributed headless mode:

- do not create orchestrator
- do not create stage routing layer
- only launch local replicas
- register with head
- perform handshake
- report liveness / queue status

This explicit split is important because it removes the need to repeatedly ask at runtime:

- "am I headless?"
- "am I distributed?"
- "do I need coordinator wiring here?"

The selected runtime mode should answer that once.

## Decoupling Coordinator-Related Information

The current code injects coordinator-related addresses and semantics into too many places. This creates strong coupling.

The refactor should make coordinator usage pluggable through interfaces.

### Suggested abstraction: `ReplicaStatusReporter`

Instead of letting stage processes know directly about `OmniCoordinator`, provide a small interface:

- `report_started(...)`
- `report_update(...)`
- `report_heartbeat(...)`
- `report_stopped(...)`

Possible implementations:

- `NullReplicaStatusReporter`
- `CoordinatorReplicaStatusReporter`

This keeps stage runtime code independent from the concrete coordinator implementation.

### Suggested abstraction: `ReplicaSnapshotSource`

Instead of letting routing logic know directly about `OmniCoordClientForHub`, provide:

- `get_replicas_for_stage(stage_id)`

Possible implementations:

- `StaticReplicaSnapshotSource`
- `CoordinatorReplicaSnapshotSource`

This keeps routing code generic and lets coordinator stay inside distributed-only wiring.

## Proposed File-Level Structure

The refactor should prefer adding new focused modules first, then gradually migrating responsibilities out of the current large files.

Suggested new modules:

- `vllm_omni/engine/deployment_plan.py`
- `vllm_omni/engine/runtime_factory.py`
- `vllm_omni/engine/launch_utils.py`
- `vllm_omni/engine/replica_registry.py`
- `vllm_omni/engine/stage_router.py`
- `vllm_omni/distributed/membership_controller.py`
- `vllm_omni/distributed/status_reporter.py`

Suggested role after migration:

- `serve.py`
  - CLI parsing
  - mode dispatch only

- `async_omni_engine.py`
  - high-level composition only

- `stage_engine_startup.py`
  - low-level registration / protocol / handshake utilities

- `orchestrator.py`
  - request flow only

- `stage_pool.py`
  - stage-facing runtime facade only

## Proposed Phased Refactor Plan

### Phase 1: Introduce a deployment plan without changing behavior

Goal:

- create a single normalized representation of runtime topology

Actions:

- add `OmniDeploymentPlan`
- compute it from existing CLI/config inputs
- move distributed mode inference out of scattered runtime code

Expected outcome:

- behavior unchanged
- topology logic becomes inspectable and testable

### Phase 2: Extract bootstrap and launch utilities

Goal:

- centralize process bring-up, registration, handshake, and readiness logic

Actions:

- add a dedicated launch/bootstrap module
- move remote registration + handshake out of `serve.py` and `async_omni_engine.py`
- keep current protocol intact initially

Expected outcome:

- `serve.py` becomes much thinner
- orchestration code stops depending on wire-level setup

### Phase 3: Introduce coordinator-neutral interfaces

Goal:

- remove direct coordinator coupling from common runtime paths

Actions:

- add `ReplicaStatusReporter`
- add `ReplicaSnapshotSource`
- replace direct coordinator usage in stage runtime and routing paths

Expected outcome:

- coordinator becomes a distributed-only plugin
- local mode stops carrying coordinator-specific plumbing

### Phase 4: Split membership from routing

Goal:

- reduce the scope of `StagePool`

Actions:

- add `ReplicaRegistry`
- add `StageRouter`
- move attach/detach and address mapping out of `StagePool`
- move affinity/LB logic out of `StagePool`

Expected outcome:

- `StagePool` becomes smaller and easier to reason about
- routing policy can evolve independently from lifecycle logic

### Phase 5: Shrink `Orchestrator`

Goal:

- keep `Orchestrator` focused on request flow only

Actions:

- move replica attach/detach handling to `MembershipController`
- remove direct remote-replica bootstrap details from `Orchestrator`

Expected outcome:

- `Orchestrator` becomes easier to test
- distributed membership changes stop affecting request-flow code directly

## Recommended First Acceptance Criteria

The first refactor stage should be considered successful if all of the following become true:

- single-node mode can run without importing coordinator-specific runtime pieces
- headless mode can run without importing orchestrator-specific logic
- `serve.py` no longer contains detailed distributed bootstrap logic
- `Orchestrator` no longer builds remote replica clients directly
- `StagePool` no longer owns both routing policy and membership lifecycle

## Practical Notes

- This refactor should prefer incremental extraction over large rewrites.
- The first goal should be structural separation, not behavioral expansion.
- The LLM path should be cleaned up first; the diffusion path can be migrated later once the abstractions are stable.
- The most important design constraint is to keep distributed-only machinery out of local request-flow code.

## Summary

The main lesson from upstream `vllm` is not a specific implementation detail, but a structural one:

- normalize topology early
- choose execution mode once
- isolate bootstrap/handshake logic
- keep coordinator as control plane
- keep orchestration focused on requests

`vllm-omni` should move toward the same shape. The single biggest improvement will come from explicitly separating:

- deployment planning
- bootstrap / control plane
- request orchestration

instead of continuing to let all three live inside the same runtime modules.
