# AgenticCV – Capsule Anomaly Classification with DSPy

![AgenticClassifcMLImage](AgenticML.png "Classic ML with Agentic AI")

This project uses **[DSPy](https://github.com/stanfordnlp/dspy)** to build an **agentic computer vision classifier** over the **MVTec AD – Capsule** dataset.

DSPy is used to:

1. Wrap a multimodal LLM (e.g. `openai/gpt-4o`) in a **Chain-of-Thought classifier**.
2. Automatically **optimize the prompt/program** with **MIPROv2**.
3. Export the optimized programs as `.json` files that capture the full configuration.

In this setup:

- The **baseline DSPy model** (unoptimized) reaches roughly **60% exact-match accuracy** on the test set.
- After **MIPROv2 (light/heavy) optimization**, the model reaches around **76% exact-match accuracy** on the same test set.

---

## Key Ideas

- Use a **folder-based image dataset** where each subfolder is a label.
- Turn every image into a `dspy.Example` with:
  - `image_path` (input)
  - `answer` (label)
- Define a **DSPy Signature** that takes an image and returns a label, with chain-of-thought reasoning.
- Use **`dspy.Evaluate` + `answer_exact_match`** to measure accuracy.
- Use **`MIPROv2`** to automatically refine the program and export the resulting configuration to JSON.

---

## Requirements

- Python 3.10+
- DSPy and dependencies:
  - `dspy-ai`
  - `pandas`
  - `scikit-learn`
  - `python-dotenv`
- A multimodal LLM backend supported by DSPy (e.g. OpenAI `gpt-4o` with vision support).

Example install:

```bash
pip install dspy-ai pandas scikit-learn python-dotenv
```

You also need environment variables configured for your LLM provider (e.g. `OPENAI_API_KEY`).

---

## Dataset

This project uses the **Capsule** subset of the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.

The code assumes a path such as:

```python
DATASET = "/.../mvtec-ad/versions/2/capsule/test"
DATASET_PATH = Path(DATASET)
```

With a folder structure like:

```text
capsule/
  test/
    crack/
      ...
    scratch/
      ...
    poke/
      ...
    faulty_imprint/
      ...
    squeeze/
      ...
    good/
      ...
```

Each subfolder name is treated as the **label** (defect type).

---

## How It Works

### 1. Build an image dataframe

```python
def build_image_dataframe(dataset_path: Path, output_csv: str | None = None) -> pd.DataFrame:
    """
    Walks a dataset folder where each subfolder is a label and contains images.
    Returns a DataFrame with:
      - path: 'label/image.png'
      - label: subfolder name
    """
    ...
```

- Walks all subdirectories under `DATASET_PATH`.
- Filters by image extensions (`.jpg`, `.jpeg`, `.png`).
- Builds a `DataFrame` with:
  - `path`: relative path like `crack/img_001.png`
  - `label`: folder name (e.g. `crack`, `good`)
- Optionally writes `dataset/data.csv`.

```python
full_df = build_image_dataframe(DATASET_PATH, output_csv="dataset/data.csv")
```

### 2. Build DSPy examples

```python
examples = [
    dspy.Example(
        image_path=str(DATASET_PATH / row["path"]),
        answer=row["label"],
    ).with_inputs("image_path")
    for _, row in full_df.iterrows()
]
```

- Converts each row into a `dspy.Example`.
- `image_path` is a string path; `answer` is the label.
- `with_inputs("image_path")` tells DSPy which field is the input.

### 3. Stratified train/test split

```python
from sklearn.model_selection import train_test_split

labels = [ex.answer for ex in examples]

train_examples, test_examples = train_test_split(
    examples,
    test_size=0.2,
    stratify=labels,
    random_state=42,
)
```

- Keeps label proportions similar in train and test.
- `train_examples` and `test_examples` are used by DSPy.

### 4. Define the DSPy Signature and Module

```python
class AnomalyDetectionCapsuleSignature(dspy.Signature):
    """
    Examine the provided image of a capsule carefully. Analyze the image to
    detect any visible defects and classify the capsule into one of the
    categories: crack, scratch, poke, faulty_imprint, squeeze, or good.
    Provide a detailed reasoning of your analysis, explaining each step that
    leads to your final classification.
    """
    image: dspy.Image = dspy.InputField(desc="path to an image to be classified")
    answer: str = dspy.OutputField(desc="classification result")


class AnomalyDetector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(AnomalyDetectionCapsuleSignature)

    def forward(self, image_path: str):
        image = dspy.Image(image_path)
        return self.classify(image=image)
```

- `AnomalyDetectionCapsuleSignature` defines the **input** (`dspy.Image`) and **output** (`str` label) plus instructions.
- `AnomalyDetector` wraps the signature in a `ChainOfThought` module for step-by-step reasoning.

### 5. Configure the LLM and evaluate the baseline

```python
dspy.configure(lm=dspy.LM(model="openai/gpt-4o"))

model = AnomalyDetector()
evaluator = Evaluate(devset=test_examples, metric=answer_exact_match)

result = evaluator(model)
print(f"Baseline exact match: {result.score:.2f}")
```

- Configures DSPy to use a multimodal LLM (here `openai/gpt-4o`).
- Evaluates the unoptimized model on the test set with **exact match**.
- In typical runs, this baseline scores around **0.60 (60%)** exact match.

### 6. Optimize with MIPROv2

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=answer_exact_match,
    auto="heavy",          # or "light" for faster / lighter optimization
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
    num_threads=8,
)

compiled_model = optimizer.compile(
    model,
    trainset=train_examples[:80],
)

compiled_model.save("output/miprov2_optimized_best.json")

result = evaluator(compiled_model)
print(f"Improved model exact match: {result.score:.2f}")
```

- `MIPROv2` automatically refines the DSPy program (prompt, reasoning steps, etc.).
- Uses a subset of `train_examples` (e.g. 80 examples) for optimization.
- Saves the compiled program as `output/miprov2_optimized_best.json`.
- Re-evaluates the optimized model on the test set.

In typical runs:

- Baseline exact match ≈ **0.60 (60%)**
- MIPROv2-optimized model ≈ **0.76 (76%)**

Exact numbers depend on the model, random seed, and subset size.

---

## Outputs

After running the script, you should see:

- `dataset/data.csv` – table with image paths and labels.
- `output/miprov2_optimized_best.json` – the **compiled DSPy program**:
  - Stores the optimized prompt, chain-of-thought structure, and configuration.
  - Can be loaded later to reuse the exact same model without re-optimizing.

You can create multiple optimized variants (e.g. with different `auto` settings or models) and save them as:

```text
output/
  miprov2_optimized_base.json
  miprov2_optimized_gpt_4o.json
  miprov2_optimized(light).json
  ...
```

Each JSON file serves as a portable, versioned snapshot of your agentic CV classifier.

---

## Running the Project

1. Install dependencies and configure your LLM API keys.
2. Adjust `DATASET` in the script to point to your local MVTec Capsule folder. You can use `dataset/download_dataset.py` script.
3. Run:

```bash
python main.py
```

You should see logs like:

- Building the dataframe, number of examples.
- Baseline evaluation with exact-match score.
- MIPROv2 optimization progress.
- Improved evaluation with a higher exact-match score.

---

## Extensions

- Compare different LLMs (e.g. `gpt-4o-mini`, `gpt-4o`, other vision models).
- Try different **Signatures** (short vs detailed instructions) and measure the effect on performance.
- Add per-class metrics or confusion matrices on top of `answer_exact_match`.
- Swap in other MVTec categories (e.g. bottle, hazelnut) to test generalization of the approach.

This project is designed as a small, focused example of **agentic CV classification with DSPy**, showcasing how prompt/program optimization can significantly improve accuracy while remaining fully inspectable via the exported JSON programs.
