# Diabetic Retinopathy Severity Grading

A deep learning project for automated **Diabetic Retinopathy (DR) severity grading** from retinal fundus images.

The goal of this project is to develop a reliable deep learning system capable of classifying retinal fundus images into different stages of diabetic retinopathy severity.

---

## Overview

Diabetic Retinopathy is one of the major complications of diabetes and can lead to serious vision loss if it is not detected and treated appropriately.

Manual grading of retinal fundus images requires expert knowledge and can be time-consuming. Deep learning provides an opportunity to assist this process by automatically analyzing retinal images and predicting the severity of diabetic retinopathy.

In this project, we investigate different deep learning approaches for **five-class diabetic retinopathy severity grading**:

- No DR
- Mild DR
- Moderate DR
- Severe DR
- Proliferative DR

The project starts with conventional convolutional neural network (CNN) baselines and gradually moves toward more advanced architectures and learning strategies.

---

## Project Goals

The main goals of the project are:

- Develop a strong baseline for diabetic retinopathy grading.
- Investigate the effect of image augmentation and training strategies.
- Explore modern architectures such as Transformer-based models.
- Combine global and local visual information using multi-branch architectures.
- Investigate ordinal and boundary-aware learning for the ordered nature of DR severity levels.
- Develop a final model that provides strong and reliable grading performance.

The project is organized around a sequence of experiments, allowing different ideas and architectural changes to be evaluated systematically.

---

## Dataset

The project uses retinal fundus images from the **APTOS Diabetic Retinopathy dataset**.

Each image is associated with one of five severity levels:

```text
0 → No DR
1 → Mild
2 → Moderate
3 → Severe
4 → Proliferative DR
````

The dataset is divided into training, validation, and unseen test sets.

The test set is kept separate from the training process and is used for final evaluation of trained models.

Expected dataset structure:

```text
data/
├── train_1.csv
├── valid.csv
├── test.csv
│
├── train_images/
├── val_images/
└── test_images/
```

The dataset itself is not included in this repository.

---

## Approach

The project follows an incremental experimental approach.

### Baseline Models

The initial experiments establish conventional CNN-based baselines and provide a reference point for later experiments.

### Data Augmentation

Different augmentation strategies are investigated to improve the model's robustness to variations in retinal fundus images.

### Transformer-based Models

Transformer-based architectures are introduced to investigate their ability to capture broader contextual information from retinal images.

### Dual-Branch Architecture

A later stage of the project explores a global-local architecture with two complementary branches:

```text
                    Fundus Image
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Global Branch          Local Branch
          Transformer               CNN
              │                     │
              └──────────┬──────────┘
                         │
                      Fusion
                         │
                Classification +
                 Ordinal Learning
```

The idea is to combine:

* **Global features** for understanding the overall retinal structure.
* **Local features** for capturing fine-grained visual patterns and lesions.

The architecture is further extended with ordinal learning because DR severity levels have a natural ordering.

---

## Experimental Organization

Experiments are maintained separately rather than continuously modifying a single configuration.

For example:

```text
configs/
├── exp001_baseline.yaml
├── exp002_...
├── exp003_...
├── exp004_...
└── exp005_proposed.yaml
```

Each experiment has its own configuration so that previous experiments remain reproducible.

The experimental process generally follows this progression:

```text
Baseline
   │
   ▼
Training Improvements
   │
   ▼
Augmentation
   │
   ▼
Modern Architectures
   │
   ▼
Dual-Branch Architecture
   │
   ▼
Ordinal / Boundary-Aware Learning
   │
   ▼
Final Proposed Model
   │
   ▼
Ablation Studies
```

Not every experiment is intended to appear in the final paper. Early experiments are mainly used for development and model selection, while the most important experiments and final ablation studies are retained for the final research results.

---

## Project Structure

```text
dr-severity-grading/
│
├── configs/
│   ├── exp001_baseline.yaml
│   ├── ...
│   └── exp005_proposed.yaml
│
├── data/
│   └── ...
│
├── outputs/
│   └── ...
│
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
├── src/
│   ├── augmentations/
│   ├── callbacks/
│   ├── datasets/
│   ├── evaluation/
│   ├── loss/
│   ├── models/
│   ├── trainer/
│   └── utils/
│
├── tests/
│
├── pyproject.toml
└── README.md
```

### Important directories

**`configs/`**

Contains the configuration files for individual experiments.

**`src/models/`**

Contains the model architectures used throughout the project.

**`src/datasets/`**

Contains dataset and DataLoader implementations.

**`src/loss/`**

Contains the different learning objectives, including classification and ordinal/boundary-aware losses.

**`src/augmentations/`**

Contains image preprocessing and augmentation strategies.

**`src/evaluation/`**

Contains evaluation utilities and visualization code.

**`src/trainer/`**

Contains the training loop and training management logic.

**`scripts/`**

Contains the main entry points for training and evaluation.

**`outputs/`**

Stores experiment-specific checkpoints, metrics, figures, and other generated results.

---

## Configuration

Experiments are controlled through YAML configuration files.

This keeps important experiment settings outside the Python source code and makes experiments easier to reproduce and compare.

A configuration typically contains sections for:

```yaml
experiment_name:
seed:

dataset:
dataloader:

model:

optimizer:
scheduler:
loss:

trainer:

augmentation:

output:
logging:
```

This allows a new experiment to be created by adding a new configuration file instead of modifying the configuration of an existing experiment.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/matinsp7/dr-severity-grading.git
cd dr-severity-grading
```

The project uses Python and `uv` for dependency management.

Install the project dependencies with:

```bash
uv sync
```

---

## Preparing the Dataset

Place the dataset according to the expected directory structure:

```text
data/
├── train_1.csv
├── valid.csv
├── test.csv
├── train_images/
├── val_images/
└── test_images/
```

The CSV files should contain image identifiers and their corresponding DR labels.

Example:

```text
id_code,diagnosis
image_001,0
image_002,2
image_003,4
```

---

## Training

To train a specific experiment, provide its configuration file:

```bash
uv run python -m scripts.train \
    --config configs/exp005_proposed.yaml
```

Different experiments can be run by selecting their corresponding configuration:

```bash
uv run python -m scripts.train \
    --config configs/exp001_baseline.yaml
```

Training results are stored under:

```text
outputs/<experiment_name>/
```

Depending on the experiment, this may include:

* Model checkpoints
* Training metrics
* Evaluation results
* Figures
* Other experiment artifacts

---

## Resuming Training

Experiments that support checkpoint resumption can be continued using:

```bash
uv run python -m scripts.train \
    --config configs/exp005_proposed.yaml \
    --resume
```

This allows interrupted training sessions to continue from an existing checkpoint.

---

## Evaluation

After training, the best saved checkpoint can be evaluated on the unseen test set:

```bash
uv run python -m scripts.evaluate \
    --config configs/exp005_proposed.yaml
```

The evaluation pipeline generates predictions and produces the corresponding evaluation results and visualizations.

Generated figures can include:

* Confusion matrix
* Normalized confusion matrix
* ROC curve
* Precision-Recall curve

---

## Experiment Tracking

Training experiments can be tracked using **Weights & Biases (W&B)**.

The project records training and validation information during experiments and can also log final test results and generated visualizations.

This makes it easier to compare different experiments without relying only on local output files.

---

## Reproducibility

Experiments are designed to be reproducible through:

* Dedicated configuration files
* Fixed random seeds
* Explicit model configurations
* Explicit optimizer and training settings
* Saved checkpoints
* Experiment-specific output directories

Each experiment should be treated as an independent configuration rather than modifying an existing experiment after it has been completed.

---

## Development Workflow

The project uses Git branches to separate experimental development.

A typical workflow is:

```text
dev
 │
 ├── design_exp001
 ├── design_exp002
 ├── design_exp003
 └── design_exp005
```

An experimental branch can be used to develop and test a specific idea before merging the stable result into the main development branch.

---

## Research Direction

The project is gradually moving from a standard image classification problem toward a more specialized approach for diabetic retinopathy grading.

A key research direction is to take advantage of the fact that DR severity is **ordinal**:

```text
No DR < Mild < Moderate < Severe < Proliferative
```

Therefore, confusing neighboring severity levels should generally be considered differently from confusing very distant levels.

The proposed direction combines:

* Global and local visual representation
* Modern deep learning architectures
* Ordinal learning
* Boundary-aware learning
* Robust training strategies

The final stages of the project focus on evaluating this proposed approach and performing targeted ablation studies to understand the contribution of its individual components.

---

## Status

The repository is under active development.

The experimental pipeline is being developed incrementally, with early experiments serving as baselines and feasibility studies and later experiments focusing on the proposed architecture and learning strategy.

The final research model and ablation studies are still under development.

---

## License

This project is intended primarily for research and educational purposes.

```
