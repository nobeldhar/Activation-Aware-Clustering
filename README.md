

# Euro-Par 2025 Artifact Overview  
**Title:** A Sparsity Predicting Approach for Large Language Models via Activation Pattern Clustering  
**Conference:** Euro-Par 2025 – European Conference on Parallel and Distributed Computing  

This artifact accompanies the paper *“A Sparsity Predicting Approach for Large Language Models via Activation Pattern Clustering,”* accepted at Euro-Par 2025. The artifact provides all necessary components to reproduce the key experimental results presented in the paper, including activation extraction, sparsity enforcement, centroid-based clustering, and perplexity evaluation.

---

## 📁 Artifact Contents

```
artifact/
├── llm-awq/                          # Modified codebase for activation extraction and evaluation
│   ├── awq/entry.py                  # Handles sparsity thresholding and PPL scoring
│   ├── env.yml                       # Conda environment specification
│   └── ...
├── clustering/                       # Clustering algorithms and generated centroids
│   ├── clustering_results_50_mistral_weighted/
│   ├── activation_aware_clustering.py
│   ├── ...
├── thresholds/
│   └── ffn_thresholds_50.json        # Thresholds to enforce 50% FFN sparsity
├── overview.pdf                      # Official Euro-Par artifact overview document
└── README.md                         # This file
```

---

## ✅ Getting Started

### 1. Environment Setup

```bash
cd llm-awq
conda env create -f env.yml
conda activate llm-awq
```

### 2. Run Perplexity Evaluation with Stored Centroids

```bash
python -m awq.entry --model_path models/Mistral-7B-v0.1/ --tasks wikitext
```

This command performs the following:
- Applies 50% sparsity to FFN layers using the provided thresholds
- Loads precomputed centroids for all three MLP sub-layers
- Executes inference and reports the final perplexity score (PPL)

---

## 🔄 Reproducing Clustering from Scratch

To regenerate clustering centroids from activation patterns:

```bash
cd clustering
python activation_aware_clustering.py
```

This script clusters activation vectors for `gate_proj`, `up_proj`, and `down_proj` across 32 layers and stores results in `clustering_results_50_mistral_weighted/`.

To evaluate model performance using these centroids, return to `llm-awq` and rerun the entry script.

---

## 🧪 Experimental Platform

- **OS:** Ubuntu 22.04  
- **GPUs:** 8 × NVIDIA A100 (80 GB each)  
- **CPU:** AMD EPYC  
- **RAM:** 512 GB  
- **CUDA:** 11.8  
- **Model:** Mistral-7B v0.1  

All experiments were conducted on the above server configuration.

---

## 📦 Outputs Included

- Centroids in `.pt` format for various cluster sizes (256–16384) and sparsity levels (20%, 30%, 40%)
- Threshold file (`ffn_thresholds_50.json`) used for enforcing sparsity
- Scripts for centroid selection, filtering, and evaluation
- Modified `llm-awq` source code for reproducibility




