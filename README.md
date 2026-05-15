# The Powerpuff Girls - Cybersecurity Recommender System

A university project implementing a cybersecurity recommender system.

## Directory Structure

```
├── src/                          # Python source code
│   ├── dashboard.py             # Main dashboard application
│   ├── dashboard123.py          # Alternative dashboard variant
│   └── train_phase3.py          # Phase 3 training script
│
├── data/                         # Data files
│   ├── exercises_full.csv       # Complete exercises dataset
│   ├── orgs_full.csv            # Complete organizations dataset
│   ├── phase2_top10_recommendations.csv
│   ├── phase3_model_comparison.csv
│   └── phase3_technique_gaps.csv
│
├── models/                       # Trained model files & artifacts
│   ├── coverage_matrix.npy
│   ├── latent_vectors.npy
│   ├── org_ids.npy
│   ├── tech_vocab.npy
│   └── test_indices.npy
│
├── training/                     # Training history & logs
│   ├── autoencoder_history.json
│   └── mlp_history.json
│
├── cache/                        # Cache files
│   └── .cache_ollama_ex3_org1.txt
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python src/dashboard.py
```

## Project Phases

- **Phase 2**: Top 10 recommendations generation
- **Phase 3**: Model comparison and technique gap analysis
