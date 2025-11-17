# Machine Learning Model Baselines with Plexe

This repository demonstrates how to use [Plexe](https://github.com/plexe-ai/plexe) to automatically build **traditional machine learning models** (e.g. logistic regression, decision trees) on the classic Breast Cancer Wisconsin dataset.

Instead of hand-writing feature engineering, training, and evaluation code, we give Plexe a high-level natural language **intent**, and its agents design and train a model using standard libraries like scikit-learn, XGBoost, etc.

Use cases:

- Quickly spin up **baseline ML models** for a tabular classification problem.
- Explore **agentic AI** applied to classic ML workflows.
- Produce a **shareable artifact** (`breast-cancer.tar.gz` or similar) containing the generated pipeline, metrics, and agent outputs.

---

## What is Plexe?

[Plexe](https://github.com/plexe-ai/plexe) is an open-source library that builds ML models from natural language prompts using a multi-agent system.

You describe your problem ("predict X from Y, use these model families, optimize for Z"), and Plexe orchestrates LLM-powered agents that:

- Analyze the dataset and intent,
- Generate training and evaluation code,
- Run experiments and iterate,
- Package the resulting model and metadata.

In this project, I used the **Python library** (not the hosted platform) so anyone can run and inspect everything locally.

---

## Project Goals

- Use **Plexe** to generate simple tabular classifiers for breast cancer diagnosis.
- Restrict the solution to **simple, fast models** (linear / tree-based, no ensembling).
- Show how agentic AI can be used to build **traditional ML baselines**.
- Provide a minimal, reproducible example that showcases:
  - Hands-on experience **using an agentic AI framework** (Plexe, built on top of `smolagents`),
  - A solid foundation in **classical ML** (logistic regression, decision trees, tabular baselines),
  - The ability to design and run **LLM-driven AutoML-style workflows** end-to-end (data prep → training → evaluation → packaging) by orchestrating an existing agent rather than hand-coding the pipeline.

---

## Repository Structure

Key files:

- `pyproject.toml`  
  Project metadata and dependencies (e.g. `plexe`, `pandas`, `scikit-learn`, `kagglehub`).

- `requirements.txt`  
  Simple requirements list if you prefer `pip install -r`.

- `download_dataset.py`  
  Helper script to download the Breast Cancer Wisconsin dataset from Kaggle via `kagglehub` and print the local path.

- `main.py`  
  Core example:
  - Loads the dataset (`dataset/data.csv`),
  - Shuffles and splits into train / test sets,
  - Calls Plexe to build the model via agents,
  - Saves the model artifact (e.g. `breast-cancer-logistic.tar.gz`),
  - Runs predictions and prints a model description.

- `breast-cancer-*.tar.gz` (generated)  
  Saved Plexe model package containing:
  - `code/` – generated training, prediction, feature transformation code,
  - `metadata/` – intent, metrics, evaluation report, run metadata,
  - `schemas/` – input/output schemas,
  - `artifacts/` – serialized model(s), e.g. `*_model.joblib`.

Example contents of the saved artifact:

```text
metadata/intent.txt
metadata/state.txt
metadata/metrics.yaml
metadata/metadata.yaml
metadata/identifier.txt
schemas/input_schema.yaml
schemas/output_schema.yaml
code/trainer.py
code/predictor.py
code/feature_transformer.py
code/dataset_splitter.py
code/testing.py
metadata/evaluation_report.yaml
artifacts/linear_regression_model.joblib
metadata/eda_report_dataset_0.md
```

## How the Example Works

All the core logic lives in `main.py`. At a high level:

1. **Load and split the dataset**

   - Read `dataset/data.csv` into a Pandas DataFrame.
   - Shuffle the rows and split into 80% train / 20% test:
     - `train_df` used to build the Plexe model.
     - `test_df` used to sanity-check predictions.
   - Optionally write `dataset/train.csv` and `dataset/test.csv` back to disk.

2. **Define the Plexe model**

   ```python
   from plexe import ModelBuilder
   from plexe.internal.common.provider import ProviderConfig

   model = ModelBuilder(
       provider=ProviderConfig(
           default_provider="openai/gpt-4o",
       ),
       verbose=False,
   )
   ```

This wires Plexe to an LLM backend (via OpenAI’s gpt-4o) and configures how the agent will be run. Under the hood, Plexe uses the smolagents framework to execute a single, LLM-powered Python agent that writes and runs the ML code for you.

3. Define the intent and build the model

The intent describes the ML problem and the constraints you want to impose:

  ```python
  m = model.build(
      datasets=[train_df],
      intent=(
          """Diagnose from the characteristics of the cell nuclei present in the
          dataset as M = malignant or B = benign.
          Use only linear regression and decision tree models, no ensembling.
          The models must be extremely simple and quickly trainable on constrained hardware."""
      ),
      output_schema={
          "diagnosis": str,
      },
      max_iterations=2,
      timeout=1800,   # 30 minutes total build time
      run_timeout=180 # 3 minutes per individual run
  )
  ```

  Plexe’s agent (implemented with smolagents) then:
  - Inspects the dataset and infers a schema,
  - Chooses appropriate preprocessing and model types,
  - Trains models within your constraints,
  - Evaluates them and converges on a simple, fast baseline.

  4. Save the model
  ```python
  import plexe
  plexe.save_model(m, "breast-cancer.tar.gz")
  ```

  This creates a portable tarball containing:

  - Generated Python code for training and prediction,
  - Model artifacts (e.g. joblib files),
  - Metadata, metrics, EDA reports,
  - Input/output schemas.

  5. Run predictions  
  ```python
  test_subset = test_df.sample(20)
  predictions = pd.DataFrame.from_records(
    [m.predict(x) for x in test_subset.to_dict(orient="records")]
  )
  print(predictions)
  ```

  This:
  - Takes a small random sample from the test set,
  - Calls the Plexe agent’s predict method row-by-row,
  - Aggregates predictions into a DataFrame for quick inspection.

  6. Inspect the model description
  ```python
  description = m.describe()
  print(description.as_text())
  ```

  This prints a human-readable summary of the pipeline the agent generated (feature handling, model choice, etc.), which is useful as a conversation piece when discussing your agentic workflow.

  7. NOTE: Patch for smolagents import check (workaround)
  Some environments trigger an "authorized imports" check inside
  `smolagents.local_python_executor`, which can incorrectly complain about
  `scikit-learn` even when installed. To avoid blocking the example,
  `main.py` includes a small patch:

  ```python
  import smolagents.local_python_executor as lpe
  def _no_check(self):
      return None
  lpe.LocalPythonExecutor._check_authorized_imports_are_installed = _no_check
  ```

  This disables that specific check so the Plexe agent can import scikit-learn and run the generated code.


  ## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>.git
cd <your-repo-folder>
```

### 2. Create and activate a virtual environment (recommended)

Use Python 3.11+:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Install dependencies

Using `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Or using `pyproject.toml`:

```bash
pip install .
```

Core dependencies include:

- `plexe` – agentic ML builder (built on top of `smolagents`)
- `pandas` – tabular data handling
- `scikit-learn` – classical ML models and utilities
- `kagglehub` – dataset download helper
- `python-dotenv` – load environment variables from `.env`

### 4. Configure environment variables

Plexe needs an LLM provider; this example uses OpenAI `gpt-4o`:

```python
provider = ProviderConfig(
    default_provider="openai/gpt-4o",
)
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-openai-key"
```

Or create a `.env` file in the project root (the script uses `load_dotenv()`):

```env
OPENAI_API_KEY=sk-...
```

---

## Downloading the Dataset

This project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset from Kaggle  
(typically `uciml/breast-cancer-wisconsin-data`).

The helper script `download_dataset.py` uses `kagglehub` to download the dataset and print the local path.

1. Make sure your Kaggle credentials are configured (per Kaggle / `kagglehub` docs).
2. Run:

   ```bash
   python download_dataset.py
   ```

3. Note the printed path and locate the main CSV file (for example `data.csv`).
4. Create the `dataset` directory (if it does not exist):

   ```bash
   mkdir -p dataset
   ```

5. Copy or symlink the CSV into the expected location:

   ```bash
   cp /path/to/kaggle/datasets/uciml/breast-cancer-wisconsin-data/data.csv dataset/data.csv
   ```

`main.py` expects the dataset to be available at:

```text
dataset/data.csv
```

---

## Running the Example

After installing dependencies, setting the environment variables, and placing the dataset in `dataset/data.csv`, run:

```bash
python main.py
```

You should see:

1. The dataset being loaded, shuffled, and split into train/test.
2. Plexe’s agent running its build process (EDA, model selection, training, evaluation) within the specified timeouts.
3. A saved model artifact in the project root, for example:

   ```text
   breast-cancer-logistic.tar.gz
   ```

4. A small DataFrame of predictions on a random sample of the test set printed to the console.
5. A textual description of the generated pipeline printed from:

   ```python
   description = m.describe()
   print(description.as_text())
   ```

You can re-run `main.py` after modifying the intent, constraints, or provider configuration to explore different agentic solutions for the same dataset.
