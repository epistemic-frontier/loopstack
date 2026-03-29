# Loopstack

**Loopstack is an agentic compiler for verifiable task loops.**

Loopstack turns a task definition into an executable, observable, and revisable loop.

A task is written in `TASK.md`, compiled into `loopstack.py`, and executed on top of the Loopstack Python library.

Instead of treating an agent as a chatbot with tools, Loopstack treats a task as a structured process:

**proposal → execution → verification → revision → memory → next proposal**

Its goal is not merely to generate outputs, but to compile tasks into loops that can be run, checked, repaired, and learned from.

---

## Why Loopstack

Large language models are useful not simply because they generate text, but because they compress vast amounts of human experience into callable priors.

However, real work is rarely solved in one shot. Useful work usually requires:

- proposing a plan,
- interacting with tools and environments,
- checking intermediate results,
- revising failed attempts,
- preserving what was learned for future runs.

The core problem is therefore not prompt writing alone.

The core problem is how to build **verifiable task loops**.

Loopstack exists to make such loops explicit, compilable, executable, and reusable.

---

## Core Idea

Loopstack takes a task written in `TASK.md`, derives a task representation from it, and compiles that representation into a runnable `loopstack.py`.

The generated `loopstack.py` is not a standalone program in spirit. It is a thin residual runtime built on top of the Loopstack Python library.

In the minimal model:

- `TASK.md` is the **source**
- `loopstack compile` produces `loopstack.py`
- `loopstack run` executes the compiled loop
- `.loopstack/` stores runtime state, traces, and memory

This gives each task a concrete execution world rather than leaving it inside an implicit chat session.

---

## Library and Compiler

Loopstack has two tightly related parts:

1. **the compiler layer**, which reads `TASK.md` and generates `loopstack.py`;
2. **the runtime library**, which provides the core semantics used by the generated loop.

In other words, `loopstack.py` is not handwritten framework glue.  
It is a compiled task runtime that depends on the Loopstack Python library for:

- task loading,
- loop execution,
- proposal handling,
- verifier orchestration,
- state transitions,
- trace recording,
- structured memory updates.

This split is intentional.

- `TASK.md` is the task source.
- `loopstack.py` is the compiled residual runtime.
- the Loopstack library provides the reusable kernel beneath it.

This design keeps the task-facing surface small while preserving a shared execution model across many tasks.

---

## Minimal Repository Spec

A minimal Loopstack repository requires only one source file:

```text
TASK.md
````

In practice, `TASK.md` contains:

1. a lightweight structured front matter, and
2. a natural-language task description.

Example:

```markdown
---
task_type: coding
max_iterations: 8
write_scope:
  - "src/**"
  - "tests/**"
verifiers:
  - "pytest -q"
  - "python -m py_compile src tests"
memory: true
---

Fix the failing tests in this repository without changing the public API.
```

The front matter provides the minimal structural constraints needed to compile a task loop.
The body describes the task itself in natural language.

---

## Command Line Interface

Loopstack provides a small command-line interface.

### `loopstack init`

Initialize a new Loopstack task repository.

```bash
loopstack init
```

This creates a minimal `TASK.md` template in the current directory.

Example generated file:

```markdown
---
task_type: coding
max_iterations: 8
write_scope:
  - "src/**"
  - "tests/**"
verifiers:
  - "pytest -q"
memory: true
---

Describe the task here.
```

---

### `loopstack compile`

Compile `TASK.md` into a runnable `loopstack.py`.

```bash
loopstack compile
```

This command:

* reads `TASK.md`,
* parses the front matter and task body,
* derives a task representation,
* generates `loopstack.py`.

The generated `loopstack.py` is the residual runtime for this task instance.
It is intentionally thin, and relies on the Loopstack Python library for the core loop machinery.

---

### `loopstack run`

Run the compiled task loop.

```bash
loopstack run
```

This command executes `loopstack.py` and starts the loop:

1. load task and state,
2. generate or refine a proposal,
3. execute actions,
4. run verifiers,
5. decide whether to accept, revise, retry, escalate, or abort,
6. update memory and traces.

If `loopstack.py` does not exist yet, implementations may either:

* require the user to run `loopstack compile` first, or
* automatically compile before execution.

The stricter initial behavior is:

```bash
loopstack compile
loopstack run
```

---

## Typical Workflow

A typical Loopstack workflow looks like this:

```bash
loopstack init
# edit TASK.md
loopstack compile
loopstack run
```

That is:

1. initialize the task repository,
2. define the task in `TASK.md`,
3. compile the task into `loopstack.py`,
4. run the compiled loop.

---

## Repository Layout

A minimal repository may look like this:

```text
TASK.md
loopstack.py
.loopstack/
```

Where:

* `TASK.md` is the task source,
* `loopstack.py` is the compiled loop runtime,
* `.loopstack/` stores runtime artifacts such as state, traces, and memory.

A more explicit layout may look like this:

```text
TASK.md
loopstack.py
.loopstack/
  state.json
  memory.json
  runs/
```

Additional files and directories may be added by particular tasks, but they are not required by the minimal spec.

---

## Mental Model

Loopstack can be understood as three layers:

* **`TASK.md`** — the task source
* **`loopstack.py`** — the compiled task-specific runtime
* **the Loopstack library** — the reusable execution kernel

The compiler lowers the task source into a thin runnable program.
The library provides the shared semantics that make the program work.

---

## What Gets Compiled

Loopstack does not merely wrap a prompt.

It compiles a task into a loop with explicit semantics:

* **task type**
  what kind of task this is

* **iteration bound**
  how many rounds are allowed

* **write scope**
  which files may be modified

* **verifiers**
  how success or failure is checked

* **memory policy**
  whether reusable structure should be preserved

* **task body**
  what the task is actually asking the system to do

In other words, Loopstack compiles not just a string, but a constrained process.

---

## Conceptual Execution Model

A Loopstack task loop typically contains the following stages:

1. **Load**
   Read `TASK.md`, runtime state, and prior memory.

2. **Propose**
   Produce a candidate plan, patch, script, or next action.

3. **Materialize**
   Turn the proposal into inspectable artifacts.

4. **Execute**
   Run tools, commands, scripts, or file updates.

5. **Verify**
   Check results using tests, rules, or human gates.

6. **Decide**
   Accept, revise, retry, escalate, or abort.

7. **Remember**
   Preserve reusable structure from the run.

This loop is the central object of the system.

---

## Verification

Verification is not an afterthought in Loopstack.
It is part of the loop itself.

In the minimal model, verifiers are declared directly in `TASK.md`, for example:

```yaml
verifiers:
  - "pytest -q"
  - "python -m py_compile src tests"
```

Over time, Loopstack may support richer verifier types, such as:

* shell-based verifiers,
* Python verifier functions,
* policy checks,
* human approval gates.

But the basic principle remains the same:

> A task loop without verification is only a sequence of guesses.

---

## Memory

Loopstack treats memory as structured task residue rather than raw chat history.

In the minimal model, memory may record:

* common failure patterns,
* effective repair patterns,
* missing context discovered during execution.

The point of memory is not archival alone.
The point is to improve future loops.

---

## Design Principles

Loopstack follows a few simple principles:

* **Task-first**
  The task definition is the source.

* **Compile, don’t just orchestrate**
  A task should be lowered into an explicit runnable loop.

* **Verification-first**
  Success must be checkable.

* **Minimal but extensible**
  Start from a small kernel and grow only when necessary.

* **Structured memory over raw logs**
  Preserve reusable patterns, not just transcripts.

* **Concrete task worlds**
  A task should run against a real repository, not float in a chat window.

---

## Scope

Loopstack is currently focused on one question:

> How can a task be compiled into a minimal, verifiable, executable loop?

This means the current scope emphasizes:

* `TASK.md` as the task source,
* a small compilation step,
* a runnable Python loop,
* explicit verifiers,
* runtime traces and memory,
* repo-based technical tasks.

Loopstack is **not** trying to be:

* a generic chatbot wrapper,
* a prompt collection,
* a no-code workflow builder,
* a project management tool with AI features.

---

## Near-Term Direction

The near-term goal of Loopstack is to define:

* a minimal `TASK.md` spec,
* a small Python runtime library,
* a compiler from task source to `loopstack.py`,
* structured execution traces,
* a first generation of verifier and memory primitives.

The first strong use case is likely to be **repo-based technical tasks**, especially tasks with rich verification surfaces such as code, scripts, experiments, and documentation workflows.

---

## Long-Term Direction

Loopstack aims to grow toward a broader agentic process system:

* richer task source formats,
* stronger task IR and compilation passes,
* reusable standard libraries for verification, policy, and memory,
* domain-specific loop backends,
* stronger observability and governance for agentic work.

The long-term ambition is not merely to automate tasks, but to make task-solving processes:

* **verifiable**
* **revisable**
* **reusable**

---

## Status

Loopstack is in an early stage.

At this stage, the project is primarily about:

* clarifying the core abstractions,
* building a small but coherent runtime,
* testing loop semantics on real tasks,
* learning from concrete execution traces.

---

## Philosophy in One Sentence

Loopstack treats a task not as a prompt, but as a verifiable loop.

---

## License

MIT License
