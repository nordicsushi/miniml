# mini-ml Framework — Architecture & Design Summary

_Last updated: 2025-11-13_

## 0) Why this framework?

We want a **small, pythonic ML pipeline framework** that:

- Treats each **Component** as a _black box_ with a clear **IO contract** (inspired by AzureML v2 Components).
- Lets users **compose Components** into a **Pipeline** (DAG at the component level), without any global step orchestration or magical DSL.
- Supports **partial execution**:
  - Within a **Component**: `start_step → end_step`
  - Within a **Pipeline**: `start_component → end_component`
  - For mid-graph pipeline runs, allow **explicit injection** of upstream outputs via `provided_outputs` (no global cache).
- Uses **Pydantic** models for specs and contracts, and keeps **step-level caching** pluggable via an `ArtifactStore`.

---

## 1) High-level Architecture

```
┌────────────┐        ┌────────────┐        ┌────────────┐
│ ComponentA │  --->  │ ComponentB │  --->  │ ComponentC │
└─────┬──────┘        └─────┬──────┘        └─────┬──────┘
      │   (component-level DAG only)         │
      ▼                                       ▼
  internal steps (A)                    internal steps (C)
```

**Key properties**

- **No global step DAG.** Each `Component` owns its internal step DAG.
- The **Pipeline** orchestrates **Components only**. Data flows via explicit bindings.
- Inputs and outputs are validated against a **ComponentContract** using **Pydantic**.
- Execution is **deterministic and explicit**: no function-call interception, no implicit graph building.

---

## 2) Core Concepts

### 2.1 Step (internal node)

- A `Step` is a Python function node inside a `Component` DAG.
- Declared with `@component.step(name=..., depends_on=[...])`.
- **Input shape**: positional args = resolved upstream step outputs; keyword args = **injected** component inputs (via contract mapping).  
- Not visible to the Pipeline; fully private to a Component.

### 2.2 Component

A `Component` is a **self-contained DAG** of steps and a **public IO contract**.

- **Contract** (`ComponentContract`):
  - `inputs: dict[str, InputSpec]`
  - `outputs: dict[str, OutputSpec]`
  - `input_to_step: dict[str, str]` → _which internal step receives which component input as kwargs_
  - `output_from_step: dict[str, str | tuple[str, str | None]]`
    - `"score": "metric"` → full step return
    - `"train_ds": ("do_split", "train_ds")` → pick a **field** from a dict returned by the step
- **Execution**: `component.run(start_step=None, end_step=None, inputs=None, use_cache=True) -> dict[str, Any]`
  - Build internal **segment** by intersecting downstream from `start_step` and upstream of `end_step`.
  - **Dependency resolution** (depth-first):
    1. If dependency in _in-memory cache_ (may hold historical artifact), reuse.
    2. If dependency in current segment, recursively execute.
    3. Else, try **artifact store** for historical result.
    4. If still missing → error (explicitness over magic).
  - **Input injection**: Component inputs are **kwargs** to the **designated step(s)** (from `input_to_step`), **not** fake step outputs.
  - **Outputs assembly**: Gather values from `output_from_step` mapping (full value or selected field), return a dict aligned with contract.

- **Caching**: Pluggable via `ArtifactStore` (default: `InMemoryArtifactStore`). Saved per `component_name + step_name`.

### 2.3 IO Specs

- `InputSpec`: `name`, `type`, `optional`, `default`, `description?`  
  - `optional + default=None` is allowed → the Component may accept `None` and decide behavior.
- `OutputSpec`: `name`, `type`, `description?`

_All specs are **Pydantic** models (clean validation and future extensibility)._

### 2.4 Pipeline (component-level DAG only)

- Nodes are `PipelineNode`s:
  - `name`: node alias in pipeline
  - `component`: the `Component` instance
  - `input_bindings`: **how to feed component inputs** (per input name) with one of:
    - `PipelineInputRef(name="...")` → comes from **PipelineInputSpec**
    - `UpstreamOutputRef(node="X", output="y")` → from **another node's outputs**
    - `ConstantValue(value=...)` → literal
- `PipelineInputSpec` declares top‐level configurable parameters (**the pipeline’s “panel of knobs”**).  
  Examples: `dataset_path: Optional[str]=None`, `debug: bool=False`, `train_ratio: float` (required), `seed: int=42`, `threshold: float=0.05`.
- **Execution**: `pipeline.run(pipeline_inputs=None, start_component=None, end_component=None, provided_outputs=None)`
  - Build component **segment** similarly to Component (downstream & upstream intersection).
  - Topological order within the segment.
  - Resolve each node’s bound inputs:
    - `PipelineInputRef`: use `pipeline_inputs[name]` or, if missing, use `PipelineInputSpec.default`; if `default is None` and `optional=True`, pass `None`; otherwise error.
    - `UpstreamOutputRef`: if upstream in current run context, use it; if upstream is **outside** the segment, consult `provided_outputs[(node, output)]`; otherwise error.
    - `ConstantValue`: use the literal.
  - Execute node via `node.component.run(inputs=resolved_inputs)`.
  - **Return value**:
    - If `end_component` specified → return that node’s outputs.
    - Else → return a dict of **sink nodes**’ outputs within the segment.

### 2.5 Bindings

- `PipelineInputRef`: binds a pipeline-level value into a component input.
- `UpstreamOutputRef`: binds one node’s **named output** into another node’s input.
- `ConstantValue`: binds a literal.

---

## 3) Error Handling (philosophy)

- **Explicitness** over silent magic. Fail with actionable messages when:
  - A component input has no binding and is required with no default.
  - An upstream output is referenced but not available in the current segment and not provided via `provided_outputs`.
  - A step depends on a missing value and there is no way to compute or cache it.
  - A contract mapping is incomplete (`input_to_step` / `output_from_step` missing).

---

## 4) Example Walkthrough

### 4.1 Components

- **prep**: `load → normalize → summarize`  
  Inputs: `dataset_path?: str=None`, `debug?: bool=False`  
  Outputs: `dataset` (from `normalize`), `meta` (from `summarize`)

- **split**: `ingest → do_split`  
  Inputs: `dataset`, `train_ratio` (required), `seed?: int=42`  
  Outputs: `train_ds` (from `("do_split", "train_ds")`), `test_ds` (field select)

- **train**: `ingest → fit`  
  Inputs: `train_ds`, `learning_rate?: float=0.1`  
  Output: `model` (from `fit`)

- **evaluate**: `ingest → metric → check`  
  Inputs: `model`, `threshold?: float=0.05`, `debug?: bool=False`  
  Outputs: `score` (metric), `passed` (check)

### 4.2 Pipeline

Inputs (`PipelineInputSpec`):  
`dataset_path?: str=None`, `debug?: bool=False`, `train_ratio: float (required)`, `seed?: int=123`, `threshold?: float=0.05`

Bindings overview:

- `prep.dataset_path ← PipelineInputRef("dataset_path")`
- `prep.debug       ← PipelineInputRef("debug")`
- `split.dataset    ← UpstreamOutputRef("prep", "dataset")`
- `split.train_ratio← PipelineInputRef("train_ratio")`
- `split.seed       ← PipelineInputRef("seed")`
- `train.train_ds   ← UpstreamOutputRef("split", "train_ds")`
- `train.learning_rate ← ConstantValue(0.2)`
- `evaluate.model   ← UpstreamOutputRef("train", "model")`
- `evaluate.threshold ← PipelineInputRef("threshold")`
- `evaluate.debug   ← PipelineInputRef("debug")`

### 4.3 Partial Runs

- **Component subgraph**:  
  `comp_prep.run(start_step="normalize", end_step="normalize", inputs={...})`
- **Pipeline subgraph**:  
  `pipe.run(start_component="split", end_component="evaluate", pipeline_inputs={...}, provided_outputs={( "prep", "dataset" ): <value>})`

---

## 5) Key Design Choices & Trade-offs

1. **Component-level black boxes** (vs global step DAG):
   - ✅ Simple mental model; strong encapsulation; easy reuse & team ownership.
   - ❌ Can’t do global cross-step optimization; need explicit `provided_outputs` for mid-graph runs.

2. **Explicit IO contract** (Pydantic specs):
   - ✅ Validates inputs/outputs, documents interfaces, eases UI/CLI generation.
   - ❌ Requires authors to think about interface design up front.

3. **No function-call interception DSL** (vs Dagster/Prefect-style call tracing):
   - ✅ Predictable, debuggable, plain Python.
   - ❌ Slightly more wiring (bindings) when composing.

4. **Caching only inside Component** (step artifacts):
   - ✅ Keeps boundaries clean; you decide where/how to store artifacts.
   - ❌ For pipeline-level reuse, caller must pass `provided_outputs` or implement a pipeline artifact store (future work).

---

## 6) Module Overview

```
miniml/
  exceptions.py      # Typed errors (InvalidSegmentError, StepNotFoundError, etc.)
  io.py              # InputSpec, OutputSpec  (Pydantic)
  bindings.py        # PipelineInputRef, UpstreamOutputRef, ConstantValue
  step.py            # Step (internal to Component)
  artifacts.py       # ArtifactStore, InMemoryArtifactStore
  component.py       # Component + ComponentContract (Pydantic)
  pipeline.py        # Pipeline, PipelineNode, PipelineInputSpec (Pydantic)
examples/
  pipeline_inputs_showcase.py
```

---

## 7) Public API (minimal)

### Component

```python
comp = Component(name="...", contract=ComponentContract(...))

@comp.step(name="load")
def load(...): ...

@comp.step(name="normalize", depends_on=["load"])
def normalize(load_out): ...

# IO contract helpers
comp.set_inputs({...}, input_to_step={...})
comp.set_outputs({...}, output_from_step={...})

# Execute a subgraph and return outputs by contract key
outs = comp.run(
  start_step: str | None = None,
  end_step: str | None = None,
  inputs: dict[str, Any] | None = None,
  use_cache: bool = True,
) -> dict[str, Any]
```

### Pipeline

```python
pipe = Pipeline(name="...")
pipe.add_input(PipelineInputSpec(...))  # top-level params

pipe.add_node(PipelineNode(
  name="prep",
  component=comp_prep,
  input_bindings={
    "dataset_path": PipelineInputRef(name="dataset_path"),
  },
))

# Optional: link helper
pipe.link("prep", "dataset", "split", "dataset")

# Execute
result = pipe.run(
  pipeline_inputs: dict[str, Any] | None = None,
  start_component: str | None = None,
  end_component: str | None = None,
  provided_outputs: dict[tuple[str, str], Any] | None = None,
)
```

---

## 8) Future Improvements

1. **Pipeline-level Artifact Store**  
   - Store/retrieve node outputs across runs; eliminate manual `provided_outputs` for mid-graph runs.

2. **Richer Output Mapping**  
   - Support nested field selection with dotted paths (e.g., `"foo.bar[0]"`).

3. **Type System & Schema Checking**  
   - Stronger type compatibility checks between bindings (producer output vs consumer input spec).

4. **Visualization**  
   - `component.visualize()` and `pipeline.visualize()` (Graphviz/NetworkX) with drill-down from component to step DAG.

5. **Execution Backends**  
   - Allow components to register execution backends (local, Docker, remote service) while keeping the same API.

6. **Retry/Timeout/Logging Policies**  
   - Step-level or component-level execution policies (retries, timeouts, structured logs).

7. **Better Defaults for Input Injection**  
   - Allow mapping `{component_input: (step_name, param_name)}` when step param names differ from input names.

8. **Versioning & Reproducibility**  
   - Add `run_id`, input hashing, and artifact lineage for reproducible pipelines.

9. **Simple UI or CLI**  
   - Auto-generate CLI from `PipelineInputSpec`; minimal web UI to configure inputs and run pipelines.

---

## 9) Quick FAQ

- **Why not infer dependencies automatically from code?**  
  We prefer explicit, predictable wiring with clear contracts, which scales better across teams and environments.

- **Can I re-run a single Step?**  
  We support **Component subgraphs** (start/end step) and cache within a component. For Pipeline-level re-runs, you can run a mid-graph subgraph and inject required upstream outputs.

- **How do I pass constants vs parameters?**  
  Use `ConstantValue` for literals; use `PipelineInputRef` to consume a pipeline parameter with default/required semantics.

---

_This document summarizes the current implementation and the intended evolution path. It should help onboard collaborators and guide future refactors without breaking the public API._