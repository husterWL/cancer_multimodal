# KAML

This repository contains code and example inputs for the KAML breast lesion classification study.

## Quick Start

Python 3.9 or later is recommended. Install the core dependencies and run the synthetic example from this directory:

```bash
python -m pip install -r requirements.txt
python demo.py
```

The demo trains the KAML model on synthetic features, writes predictions and attention outputs under `outputs/demo/`, and checks checkpoint reload consistency. The included 20-subject example is entirely synthetic; it contains no patient records or histology images and is intended only to exercise the code.

To run the five-fold KAML training entry point on the synthetic feature layout:

```bash
export KAML_OUTPUTS_ROOT="$PWD/sample_data/outputs_layout"
export KAML_RUN_DIR="$PWD/outputs/kaml"
python models/KAML/train.py
```

The trainer runs 70 epochs per fold. Use a fresh output directory for a new run: completed folds are skipped. The synthetic results are not study results. Other model entry points are under `models/`; quote paths containing spaces.

```bash
python models/SVM/train.py --csv sample_data/clinical.csv --output outputs/svm
python evaluate.py outputs/kaml/14.2/oof_predictions.csv --compare outputs/svm/oof_predictions.csv --output outputs/evaluation.json
```

To regenerate the synthetic inputs, run `python make_sample_data.py`. For external feature data, follow the directory layout and array keys in `sample_data/outputs_layout/`. Clinical CSV files contain `patient_id,label,fold` and categorical feature columns. Evaluation CSV files contain one held-out row per patient with `patient_id,fold,label,probability`; label 1 denotes malignancy.

## Repository Contents

- `models/`: KAML, baseline and ablation training code.
- `knowledge/`: clinical/pathology graph data and frozen concept-query vectors.
- `visualization/`: code for rendering concept-to-patch figures from external image inputs.
- `sample_data/`: synthetic feature data and example fold layout.
- `demo.py`, `make_sample_data.py`, `evaluate.py`: sample run, sample generation and OOF evaluation.

## Pretrained Weights

Pretrained encoder weights are not included. Obtain the required model files from their official sources and configure local paths as described in the model entry points.

- PLIP: [Hugging Face model page](https://huggingface.co/vinid/plip)
- CLIP ViT-B/16: [Hugging Face model page](https://huggingface.co/openai/clip-vit-base-patch16)
- MedCLIP: [Official MedCLIP repository](https://github.com/RyanWangZf/MedCLIP)
- Medical CLIP (MedICaT-ROCO): [Hugging Face model page](https://huggingface.co/luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO)
- PubMedBERT: [Hugging Face model page](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)

Check each provider's license and download instructions before use. Encoder checkpoints are not interchangeable; use the model variant and preprocessing expected by the selected code.

Install optional extraction dependencies with `python -m pip install -r requirements-encoders.txt`. The supplied feature extractor reads local model directories from `KAML_PATH_04` (PLIP), `KAML_PATH_05` (CLIP), and `KAML_PATH_06` (medical CLIP). Image figure rendering requires external original images and coordinate features; run `python visualization/render_patient_figure.py --help` for its arguments.