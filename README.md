# Matryoshka Representation Learning with Sentence Transformers

This repository provides scripts and utilities for training and evaluating sentence embedding models using [Matryoshka Representation Learning (MRL)](https://github.com/RAIVNLab/MRL/blob/main/MRL.py) with the [sentence-transformers](https://www.sbert.net/) library. The code supports multiple datasets and tasks, including NLI, STS, QQP, and Wiki IR, and leverages MRL losses and evaluation strategies to produce multi-resolution sentence embeddings.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Training Scripts](#training-scripts)
- [Preprocessing](#preprocessing)
- [Usage](#usage)

---

## Features

- **Matryoshka Representation Learning**: Train models that produce embeddings at multiple resolutions.
- **Multi-task Support**: Scripts for NLI, STS, QQP, and Wiki IR tasks.
- **Flexible Model Selection**: Easily switch between different transformer models.
- **Comprehensive Evaluation**: Includes binary classification, triplet, and information retrieval evaluators.

---

## Requirements

Install dependencies with:

```sh
pip install -r requirements.txt
```

### Required packages:
- sentence-transformers==3.0.1
- datasets
- pandas
- scikit-learn
- vnlp

## Project Structure
- `mrl-nli-train.py`: Training script for NLI datasets.
- `mrl-sts-train.py`: Training script for Semantic Textual Similarity.
- `mrl-qqp-train.py`: Training script for Quora Question Pairs.
- `mrl-wiki-train.py`: Training script for Wiki IR tasks.
- `preprocess.py`: Text preprocessing utilities.
- `requirements.txt`: Python dependencies.
- `data/`: Place your dataset files here.

## Datasets
The scripts use the following datasets (downloaded automatically or expected in `data/`):
- **NLI**: emrecan/all-nli-tr (via [HuggingFace Datasets](https://huggingface.co/datasets/emrecan/all-nli-tr))
- **STS**: emrecan/stsb-mt-turkish (via [HuggingFace Datasets](https://huggingface.co/datasets/emrecan/stsb-mt-turkish))
- **QQP**: embedding-data/QQP_triplets (via [HuggingFace Datasets](https://huggingface.co/datasets/embedding-data/QQP_triplets))
- **Wiki**: wikimedia/wikipedia (via [HuggingFace Datasets](https://huggingface.co/datasets/wikimedia/wikipedia))

## Usage
### Prepare datasets
- Assuming the datasets defined above are placed in the `data/` directory, or you can just use `load_dataset` module form Huggingface (Do not forget to make changes in the code for this case)
### Training Scripts
To train a model, run one of the following commands:

```sh
python mrl-nli-train.py
# or
python mrl-sts-train.py
# or
python mrl-qqp-train.py
# or
python mrl-wiki-train.py
```

### Outputs
Models and checkpoints will be saved in the respective output directories specified in each script.