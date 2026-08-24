# Pep2Net

A multi-label peptide property prediction framework based on a dual-branch deep learning architecture. Pep2Net combines pre-trained ESM protein language model embeddings with handcrafted physicochemical features to predict 51 functional and biological properties of peptide sequences.

## Overview

Peptide property prediction is challenging due to highly imbalanced labels and limited labeled data. Pep2Net addresses this with:

- **Dual-branch architecture**: semantic branch (ESM) + physicochemical branch (PC6, AAindex, BLOSUM62, AAC, DPC, PseAAC)
- **Cross-attention fusion**: bidirectional alignment between the two branches
- **Low-rank bilinear pooling**: for modeling cross-branch interactions
- **Multi-sample dropout head**: ensemble of 6 parallel dropout samples for robust predictions
- **Advanced training**: R-Drop, Supervised Contrastive Learning, EMA

## Predicted Properties (51 labels)

Includes antimicrobial (anti-*E. coli*, anti-*S. aureus*, antibiotic), antiviral (anti-SARS-CoV-2, anti-HIV), anti-cancer, anti-inflammatory, anti-aging, and 41 other functional categories.

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- Hugging Face Transformers (for ESM model)

```bash
pip install torch transformers pandas numpy scikit-learn iterstrat
```

## Dataset

Place `peptide.csv` in the `data/` directory. The CSV must contain:

| Column | Description |
|--------|-------------|
| `sequence` | Amino acid sequence string |
| 51 label columns | Binary (0/1) for each peptide property |

The bundled dataset contains 3,305 unique labeled peptide sequences after sequence normalization and duplicate removal.

## ESM Pre-trained Model

Download an ESM model from Hugging Face and update the path in `Task/Task.py`:

```python
pretrained_model_dir = "/path/to/your/esm_model"
```

Recommended: `facebook/esm2_t6_8M_UR50D` (small) or `facebook/esm2_t12_35M_UR50D`.

## Usage

```bash
cd Task/
python Task.py
```

This runs 10 independent training runs by default (`NUM_RUNS = 10`). Each run produces:

| Output | Description |
|--------|-------------|
| `best_model_{name}_run{N}.bin` | Best model weights |
| `best_model_{name}_run{N}.json` | Best validation metrics |
| `{name}_run{N}.log` | Training log |
| `train_run{N}.csv` / `validation_run{N}.csv` / `test_run{N}.csv` | Per-run 70%/10%/20% data splits |

## Key Configuration (`Task/Task.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Training batch size |
| `max_sen_len` | 200 | Max sequence length (tokens) |
| `learning_rate` | 5e-5 | Base learning rate |
| `epochs` | 100 | Max training epochs |
| `weight_decay` | 0.35 | L2 regularization strength |
| `rdrop_alpha` | 0.5 | R-Drop KL loss weight |
| `supcon_lambda` | 0.05 | Supervised contrastive loss weight |
| `p_span_drop` | 0.15 | Span masking augmentation probability |
| `p_blosum_sub` | 0.01 | BLOSUM substitution augmentation probability |

## Architecture

```
Input Sequence
    │
    ├── ESM Backbone ──────────────────────┐
    │   (pre-trained protein LM)           │
    │                                      ▼
    └── Physicochemical Branch      Cross-Attention Fusion
        ├── Residue-level (38-dim)         │
        │   PC6 + AAindex + BLOSUM62       ▼
        └── Global (470-dim)       Low-Rank Bilinear Pooling
            AAC + DPC + PseAAC             │
                │                          │
                └──── CNN + BiLSTM ────────┘
                                           │
                               Multi-Sample Dropout Head
                                           │
                               GroupWise Linear Classifier
                                           │
                                  51 Property Predictions
```

## Evaluation Metrics

**Per-class**: Precision, Recall, Specificity, F1, AUC, Accuracy, MCC

**Multi-label**:
- Aiming: sample-wise precision
- Coverage: sample-wise recall
- Accuracy: Jaccard similarity
- Absolute True: exact match ratio

The best model checkpoint is selected on the validation set by maximizing `aiming + coverage + accuracy`. The test set is evaluated once after training using the selected checkpoint.

Sequences are normalized and deduplicated before splitting. Duplicate sequences with identical labels are retained once, conflicting duplicate annotations stop preprocessing, and the generated training, validation, and test splits are verified to have zero sequence overlap.

## Training Techniques

| Technique | Purpose |
|-----------|---------|
| R-Drop | KL divergence between two dropout-varied forward passes |
| Supervised Contrastive | Pull together embeddings of co-labeled samples |
| EMA (decay=0.999) | Smoother parameter averaging |
| Focal Dice Loss | Handle class imbalance with effective-sample weighting |
| Multilabel Stratified Split | Preserve label distribution across train/validation/test |
| Sequence-level Deduplication | Prevent identical peptide sequences from appearing in multiple splits |

## Project Structure

```
Pep2Net/
├── data/
│   └── peptide.csv          # Main dataset
├── models/
│   └── Pep2Net_Model.py     # Core model definition
├── Task/
│   └── Task.py              # Training entry point
└── utils/
    ├── data_helpers.py      # Dataset loading and tokenization
    ├── evaluation.py        # Multi-label evaluation metrics
    └── log_helper.py        # Logging utilities
```

## Citation and archived version

If you use Pep2Net, please cite the associated manuscript and the software release. Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

- Source code: https://github.com/xiaoyao00a/Pep2Net
- Current release: [`v1.2.0`](https://github.com/xiaoyao00a/Pep2Net/releases/tag/v1.2.0)
- Current DOI archive: [https://doi.org/10.5281/zenodo.22085358](https://doi.org/10.5281/zenodo.22085358)
- Previous archived releases: [`v1.1.0`](https://github.com/xiaoyao00a/Pep2Net/releases/tag/v1.1.0), [https://doi.org/10.5281/zenodo.22084154](https://doi.org/10.5281/zenodo.22084154); [`v1.0.0`](https://github.com/xiaoyao00a/Pep2Net/releases/tag/v1.0.0), [https://doi.org/10.5281/zenodo.22054386](https://doi.org/10.5281/zenodo.22054386)

## License

Pep2Net is released under the [MIT License](LICENSE), an OSI-approved open-source license.

