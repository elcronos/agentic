# Agentic AI & GenAI Playground

This repository is a collection of focused projects that showcase how to use **agentic AI** and **GenAI** in practical, real-world settings.

The goal is to demonstrate:

- How LLM-powered agents can be used to build and run ML workflows end-to-end.
- How GenAI can be embedded into existing data/ML stacks (e.g. scikit-learn, PyTorch, traditional analytics).

Each folder in this repo is an independent project with its own `README.md`, code, and environment setup.

---

## Projects

### 1. AgenticClassicML

**Folder:** `AgenticClassicML/`  
**Readme:** [`AgenticClassicML/README.md`](AgenticClassicML/README.md)

**What it is**

An example that uses **[Plexe](https://github.com/plexe-ai/plexe)** (built on top of `smolagents`) to automatically build **classical ML baselines**.

Instead of manually writing the entire pipeline, the project:

- Uses a LLM-powered multi-agent  to:
  - Inspect a real tabular dataset,
  - Generate training and evaluation code in Python,
  - Train simple models (logistic regression, decision trees),
  - Package the pipeline and artifacts into a reusable tarball.
- Constrains the solution to **simple, interpretable models** to create strong, transparent baselines.
- Demonstrates an **LLM-driven AutoML-style workflow**:
  - Data loading and splitting,
  - Agent-driven model building,
  - Prediction and inspection,
  - Artifact creation and reuse.

**Why it matters**

This project is designed to signal:

- Hands-on experience **using agentic frameworks** (Plexe + smolagents) rather than just talking about them.
- A solid foundation in **classic ML** (logistic regression, decision trees, tabular baselines).
- The ability to design and run **LLM-orchestrated workflows** end-to-end (data prep → training → evaluation → packaging).

For details, setup instructions, and code walkthrough, see the dedicated readme:  
➡️ [`AgenticClassicML/README.md`](AgenticClassicML/README.md)

---

## Structure

As more projects are added, this repo will group them by theme, for example:

- `AgenticClassicML/` – Agent-driven classical ML baselines (Plexe, smolagents, scikit-learn).
- `AgenticXXXX/` – Agentic workflows for other modalities (e.g. image, text, graphs, retrieval, etc.).

Each project is:

- **Self-contained** – its own environment and instructions.
- **Reproducible** – clear setup and run steps.
- **Explainable** – focused readme and rationale for design choices.

---

## How to Use This Repo

- Browse the project folders to find the scenario you care about.
- Start with `AgenticClassicML` if you want a concrete, tabular ML example using Plexe.
- Use these projects as:
  - Portfolio pieces,
  - Starting points for your own agentic workflows,
  - Conversation drivers with teams building Agentic AI / GenAI products.

More projects will be added over time, covering different data types, tools, and deployment patterns.
