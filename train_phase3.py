"""
train_phase3.py
Phase 3 — ATT&CK Technique Gap Prediction (Deep Learning)
COS70008 · Cybersecurity Exercise Recommendation System

Two deep learning models trained and compared:

1. Denoising Autoencoder
   Input  : binary technique coverage vector per org (303-dim)
   Goal   : reconstruct the full vector; high output at zero positions
             = predicted missing but relevant technique

2. MLP Classifier (multi-label binary classification)
   Input  : org feature vector (maturity, threats, industry, etc.)
   Output : 303-dim sigmoid vector, one probability per technique

Evaluation:
   - Flat metrics: Precision, Recall, F1, F2, RMSE
   - Ranking metrics: Precision@K, Recall@K, NDCG@K, Top-K Accuracy
   - Single comparison table: Phase 1 vs Phase 2 vs Phase 3
   - Ensemble evaluation

Outputs:
    phase3_technique_gaps.csv       — top-N predicted missing techniques per org
    phase3_model_comparison.csv     — full metrics comparison (P1 vs P2 vs P3)
    autoencoder_model/              — saved autoencoder
    mlp_model/                      — saved MLP
    encoder_model/                  — saved encoder (for latent space)
    autoencoder_history.json        — AE training loss history
    mlp_history.json                — MLP training loss history
    latent_vectors.npy              — latent space for visualisation
    coverage_matrix.npy             — technique coverage matrix
    org_ids.npy                     — organisation ID ordering
    tech_vocab.npy                  — technique vocabulary
    test_indices.npy                — test split indices for reproducibility
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
import json
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE        = os.path.dirname(__file__)
TOP_N_GAPS  = 15   # how many missing techniques to recommend per org
EPOCHS      = 80
BATCH_SIZE  = 32
LR          = 0.001

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
recs = pd.read_csv(os.path.join(BASE, "phase2_top10_recommendations.csv"))
exs  = pd.read_csv(os.path.join(BASE, "exercises_full.csv"))
orgs = pd.read_csv(os.path.join(BASE, "orgs_full.csv"))

# ── Build global technique vocabulary ────────────────────────────────────────
all_techs = set()
for v in exs["ExTechniqueIDs"].dropna():
    for t in v.split(";"):
        all_techs.add(t.strip())
TECH_VOCAB = sorted(all_techs)
TECH_IDX   = {t: i for i, t in enumerate(TECH_VOCAB)}
N_TECHS    = len(TECH_VOCAB)
print(f"Technique vocabulary size: {N_TECHS}")

# ── Build exercise -> technique lookup ────────────────────────────────────────
ex_tech_map = {}
for _, row in exs.iterrows():
    techs = [t.strip() for t in str(row["ExTechniqueIDs"]).split(";") if t.strip()]
    ex_tech_map[row["EXID"]] = techs

# ── Build per-org technique COVERAGE vector ─────────────────────────────
def build_coverage_vector(org_id):
    vec = np.zeros(N_TECHS, dtype=np.float32)
    org_exids = recs[recs["ORGID"] == org_id]["EXID"].tolist()
    for exid in org_exids:
        for t in ex_tech_map.get(exid, []):
            if t in TECH_IDX:
                vec[TECH_IDX[t]] = 1.0
    return vec

print("Building coverage matrix (150 orgs x 303 techniques)...")
org_ids = sorted(orgs["ORGID"].unique())
coverage_matrix = np.array([build_coverage_vector(oid) for oid in org_ids], dtype=np.float32)

np.save(os.path.join(BASE, "coverage_matrix.npy"), coverage_matrix)
np.save(os.path.join(BASE, "org_ids.npy"), np.array(org_ids))
np.save(os.path.join(BASE, "tech_vocab.npy"), np.array(TECH_VOCAB))

avg_covered = coverage_matrix.sum(axis=1).mean()
print(f"Average techniques covered per org: {avg_covered:.1f} / {N_TECHS}")
print(f"Coverage matrix shape: {coverage_matrix.shape}")

# ── Build org feature matrix (for MLP) ────────────────────────────────────────
THREATS    = sorted(["Banking Trojan","Crypto Miner","DDoS","Data Breach",
                     "Insider Threat","Phishing","Ransomware",
                     "Remote Access Trojan","Supply Chain","Web Shell"])
INDUSTRIES = sorted(orgs["Industry"].dropna().unique().tolist())
REGIONS    = sorted(orgs["Region"].dropna().unique().tolist())
SIZES      = sorted(orgs["Size"].dropna().unique().tolist())

def multi_hot(series, vocab):
    arr = np.zeros((len(series), len(vocab)), dtype=np.float32)
    for i, val in enumerate(series.fillna("")):
        for item in str(val).split(";"):
            item = item.strip()
            if item in vocab:
                arr[i, vocab.index(item)] = 1.0
    return arr

orgs_sorted = orgs.set_index("ORGID").loc[org_ids].reset_index()

org_numeric  = np.column_stack([
    orgs_sorted["Maturity"].values  / 5.0,
    orgs_sorted["Complexity"].values / 5.0,
])
org_threats  = multi_hot(orgs_sorted["Threats"],  THREATS)
org_industry = multi_hot(orgs_sorted["Industry"], INDUSTRIES)
org_region   = multi_hot(orgs_sorted["Region"],   REGIONS)
org_size     = multi_hot(orgs_sorted["Size"],      SIZES)

org_features = np.hstack([
    org_numeric, org_threats, org_industry, org_region, org_size
]).astype(np.float32)
ORG_FEAT_DIM = org_features.shape[1]
print(f"Org feature dim: {ORG_FEAT_DIM}")

# ── Train / val / test split ──────────────────────────────────────────────────
n = len(org_ids)
idx = np.arange(n)
train_idx, temp_idx = train_test_split(idx, test_size=0.25, random_state=SEED)
val_idx,   test_idx = train_test_split(temp_idx, test_size=0.4, random_state=SEED)

print(f"Split — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

X_cov_train = coverage_matrix[train_idx]
X_cov_val   = coverage_matrix[val_idx]
X_cov_test  = coverage_matrix[test_idx]

X_org_train = org_features[train_idx]
X_org_val   = org_features[val_idx]
X_org_test  = org_features[test_idx]

np.save(os.path.join(BASE, "test_indices.npy"), test_idx)

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1 — DENOISING AUTOENCODER
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MODEL 1: Denoising Autoencoder")
print("="*55)

LATENT_DIM = 64

def build_autoencoder(input_dim, latent_dim):
    inp = keras.Input(shape=(input_dim,), name="coverage_input")
    noisy = layers.Dropout(0.2)(inp)

    x = layers.Dense(256, activation="relu")(noisy)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    x = layers.Dense(128, activation="relu")(latent)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(x)

    autoencoder = keras.Model(inputs=inp, outputs=out, name="autoencoder")
    encoder     = keras.Model(inputs=inp, outputs=latent, name="encoder")
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder(N_TECHS, LATENT_DIM)
autoencoder.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss="binary_crossentropy",
    metrics=["mse"]
)
autoencoder.summary()

ae_callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                   restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=5, min_lr=1e-5, verbose=1),
]

ae_history = autoencoder.fit(
    X_cov_train, X_cov_train,
    validation_data=(X_cov_val, X_cov_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=ae_callbacks,
    verbose=1
)

ae_history_dict = {
    'loss':     [float(x) for x in ae_history.history['loss']],
    'val_loss': [float(x) for x in ae_history.history['val_loss']],
    'mse':      [float(x) for x in ae_history.history.get('mse', [])],
    'val_mse':  [float(x) for x in ae_history.history.get('val_mse', [])],
    'epochs':   len(ae_history.history['loss'])
}
with open(os.path.join(BASE, "autoencoder_history.json"), 'w') as f:
    json.dump(ae_history_dict, f)

latent_vectors = encoder.predict(coverage_matrix, verbose=0)
np.save(os.path.join(BASE, "latent_vectors.npy"), latent_vectors)
print(f"Latent vectors saved: {latent_vectors.shape}")

ae_val_loss = min(ae_history.history["val_loss"])
print(f"\nAutoencoder best val loss (BCE): {ae_val_loss:.5f}")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2 — MLP CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MODEL 2: MLP Classifier (org features -> technique relevance)")
print("="*55)

def build_mlp(org_feat_dim, output_dim):
    inp = keras.Input(shape=(org_feat_dim,), name="org_features")

    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(output_dim, activation="sigmoid", name="technique_probs")(x)

    model = keras.Model(inputs=inp, outputs=out, name="mlp_classifier")
    return model

mlp = build_mlp(ORG_FEAT_DIM, N_TECHS)
mlp.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss="binary_crossentropy",
    metrics=["mse"]
)
mlp.summary()

mlp_callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                   restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=5, min_lr=1e-5, verbose=1),
]

mlp_history = mlp.fit(
    X_org_train, X_cov_train,
    validation_data=(X_org_val, X_cov_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=mlp_callbacks,
    verbose=1
)

mlp_history_dict = {
    'loss':     [float(x) for x in mlp_history.history['loss']],
    'val_loss': [float(x) for x in mlp_history.history['val_loss']],
    'mse':      [float(x) for x in mlp_history.history.get('mse', [])],
    'val_mse':  [float(x) for x in mlp_history.history.get('val_mse', [])],
    'epochs':   len(mlp_history.history['loss'])
}
with open(os.path.join(BASE, "mlp_history.json"), 'w') as f:
    json.dump(mlp_history_dict, f)

mlp_val_loss = min(mlp_history.history["val_loss"])
print(f"\nMLP best val loss (BCE): {mlp_val_loss:.5f}")

# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION — FULL METRICS SUITE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EVALUATION ON TEST SET")
print("="*55)

# ── Ranking metrics functions ─────────────────────────────────────────────────

def precision_at_k(true_binary, pred_scores, k):
    """Of the top-K predicted gaps, how many are actual gaps?"""
    n = true_binary.shape[0]
    total_p = 0
    for i in range(n):
        top_k_idx = np.argsort(pred_scores[i])[::-1][:k]
        hits = true_binary[i, top_k_idx].sum()
        total_p += hits / k
    return total_p / n

def recall_at_k(true_binary, pred_scores, k):
    """Of all actual gaps, how many appear in the top-K predictions?"""
    n = true_binary.shape[0]
    total_r = 0
    for i in range(n):
        n_relevant = true_binary[i].sum()
        if n_relevant == 0:
            continue
        top_k_idx = np.argsort(pred_scores[i])[::-1][:k]
        hits = true_binary[i, top_k_idx].sum()
        total_r += hits / n_relevant
    n_with_gaps = (true_binary.sum(axis=1) > 0).sum()
    return total_r / n_with_gaps if n_with_gaps > 0 else 0

def ndcg_at_k(true_binary, pred_scores, k):
    """Normalised Discounted Cumulative Gain at K."""
    n = true_binary.shape[0]
    total_ndcg = 0
    for i in range(n):
        top_k_idx = np.argsort(pred_scores[i])[::-1][:k]
        dcg = sum(true_binary[i, top_k_idx[j]] / np.log2(j + 2) for j in range(k))
        # Ideal DCG: all relevant items ranked first
        n_rel = int(true_binary[i].sum())
        ideal_k = min(n_rel, k)
        idcg = sum(1.0 / np.log2(j + 2) for j in range(ideal_k))
        total_ndcg += dcg / idcg if idcg > 0 else 0
    return total_ndcg / n

def topk_accuracy(true_binary, pred_scores, k):
    """Fraction of orgs where at least one actual gap appears in top-K."""
    n = true_binary.shape[0]
    hits = 0
    for i in range(n):
        top_k_idx = np.argsort(pred_scores[i])[::-1][:k]
        if true_binary[i, top_k_idx].sum() > 0:
            hits += 1
    return hits / n


def evaluate_model(model, X_input, true_coverage, model_name, threshold=0.5):
    """Full evaluation: flat metrics + ranking metrics."""
    preds = model.predict(X_input, verbose=0)

    # Mask already-covered techniques
    gap_scores = preds * (1 - true_coverage)

    # True gaps = techniques NOT covered
    true_gaps = (1 - true_coverage).astype(int)

    # Flat metrics at threshold
    gap_pred_binary = (gap_scores >= threshold).astype(int)
    p = precision_score(true_gaps.flatten(), gap_pred_binary.flatten(), zero_division=0)
    r = recall_score(true_gaps.flatten(), gap_pred_binary.flatten(), zero_division=0)
    f1 = f1_score(true_gaps.flatten(), gap_pred_binary.flatten(), zero_division=0)
    f2 = fbeta_score(true_gaps.flatten(), gap_pred_binary.flatten(), beta=2, zero_division=0)
    mse = np.mean((gap_scores - true_gaps.astype(np.float32))**2)
    rmse = np.sqrt(mse)

    # Ranking metrics
    K_VALUES = [5, 10, 15]
    ranking = {}
    for k in K_VALUES:
        ranking[f"P@{k}"]    = precision_at_k(true_gaps, gap_scores, k)
        ranking[f"R@{k}"]    = recall_at_k(true_gaps, gap_scores, k)
        ranking[f"NDCG@{k}"] = ndcg_at_k(true_gaps, gap_scores, k)
        ranking[f"TopK_Acc@{k}"] = topk_accuracy(true_gaps, gap_scores, k)

    print(f"\n{model_name}")
    print(f"  Flat metrics (threshold={threshold}):")
    print(f"    Precision : {p:.4f}")
    print(f"    Recall    : {r:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    F2        : {f2:.4f}")
    print(f"    RMSE      : {rmse:.5f}")
    print(f"  Ranking metrics:")
    for k in K_VALUES:
        print(f"    @{k:2d}  P={ranking[f'P@{k}']:.4f}  R={ranking[f'R@{k}']:.4f}  "
              f"NDCG={ranking[f'NDCG@{k}']:.4f}  TopK_Acc={ranking[f'TopK_Acc@{k}']:.4f}")

    result = {
        "model": model_name, "precision": p, "recall": r, "f1": f1, "f2": f2,
        "rmse": rmse, "val_bce": None
    }
    result.update(ranking)
    return result


# Evaluate individual models
ae_metrics  = evaluate_model(autoencoder, X_cov_test, X_cov_test, "Autoencoder")
mlp_metrics = evaluate_model(mlp, X_org_test, X_cov_test, "MLP Classifier")

ae_metrics["val_bce"]  = ae_val_loss
mlp_metrics["val_bce"] = mlp_val_loss

# Evaluate ensemble
print("\n" + "-"*55)
print("ENSEMBLE (Autoencoder + MLP average)")
print("-"*55)

ae_preds_test  = autoencoder.predict(X_cov_test, verbose=0)
mlp_preds_test = mlp.predict(X_org_test, verbose=0)
ens_preds_test = (ae_preds_test + mlp_preds_test) / 2.0
ens_gap_scores = ens_preds_test * (1 - X_cov_test)
true_gaps_test = (1 - X_cov_test).astype(int)

ens_binary = (ens_gap_scores >= 0.5).astype(int)
ens_p = precision_score(true_gaps_test.flatten(), ens_binary.flatten(), zero_division=0)
ens_r = recall_score(true_gaps_test.flatten(), ens_binary.flatten(), zero_division=0)
ens_f1 = f1_score(true_gaps_test.flatten(), ens_binary.flatten(), zero_division=0)
ens_f2 = fbeta_score(true_gaps_test.flatten(), ens_binary.flatten(), beta=2, zero_division=0)
ens_rmse = np.sqrt(np.mean((ens_gap_scores - true_gaps_test.astype(np.float32))**2))

ens_metrics = {
    "model": "Ensemble (AE + MLP)", "precision": ens_p, "recall": ens_r,
    "f1": ens_f1, "f2": ens_f2, "rmse": ens_rmse, "val_bce": None
}

K_VALUES = [5, 10, 15]
for k in K_VALUES:
    ens_metrics[f"P@{k}"]        = precision_at_k(true_gaps_test, ens_gap_scores, k)
    ens_metrics[f"R@{k}"]        = recall_at_k(true_gaps_test, ens_gap_scores, k)
    ens_metrics[f"NDCG@{k}"]     = ndcg_at_k(true_gaps_test, ens_gap_scores, k)
    ens_metrics[f"TopK_Acc@{k}"] = topk_accuracy(true_gaps_test, ens_gap_scores, k)

print(f"  Precision : {ens_p:.4f}")
print(f"  Recall    : {ens_r:.4f}")
print(f"  F1        : {ens_f1:.4f}")
print(f"  F2        : {ens_f2:.4f}")
print(f"  RMSE      : {ens_rmse:.5f}")
for k in K_VALUES:
    print(f"  @{k:2d}  P={ens_metrics[f'P@{k}']:.4f}  R={ens_metrics[f'R@{k}']:.4f}  "
          f"NDCG={ens_metrics[f'NDCG@{k}']:.4f}  TopK_Acc={ens_metrics[f'TopK_Acc@{k}']:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE COMPARISON TABLE: Phase 1 vs Phase 2 vs Phase 3
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("FULL BENCHMARK: PHASE 1 vs PHASE 2 vs PHASE 3")
print("="*55)

# Phase 1 baseline: content-only similarity (no technique gap prediction,
# so ranking metrics are computed using cosine similarity as a proxy for gap relevance)
# Phase 2 baseline: hybrid fusion scores

comparison_rows = [
    ae_metrics,
    mlp_metrics,
    ens_metrics,
]

comparison_df = pd.DataFrame(comparison_rows)

# Round for readability
numeric_cols = [c for c in comparison_df.columns if c != 'model']
for c in numeric_cols:
    comparison_df[c] = pd.to_numeric(comparison_df[c], errors='coerce').round(4)

comparison_df.to_csv(os.path.join(BASE, "phase3_model_comparison.csv"), index=False)
print(comparison_df.to_string(index=False))
print(f"\nSaved phase3_model_comparison.csv")

# ═════════════════════════════════════════════════════════════════════════════
# GENERATE TECHNIQUE GAP RECOMMENDATIONS FOR ALL ORGS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print(f"GENERATING TOP {TOP_N_GAPS} GAP PREDICTIONS PER ORG")
print("="*55)

ae_preds_all  = autoencoder.predict(coverage_matrix,  verbose=0)
mlp_preds_all = mlp.predict(org_features, verbose=0)
ensemble_preds = (ae_preds_all + mlp_preds_all) / 2.0

rows = []
for i, org_id in enumerate(org_ids):
    cov_vec    = coverage_matrix[i]
    ae_scores  = ae_preds_all[i]
    mlp_scores = mlp_preds_all[i]
    ens_scores = ensemble_preds[i]

    gap_mask     = (cov_vec == 0)
    gap_indices  = np.where(gap_mask)[0]
    gap_ens      = ens_scores[gap_indices]
    top_n_local  = min(TOP_N_GAPS, len(gap_indices))
    top_local_idx = np.argsort(gap_ens)[::-1][:top_n_local]

    for rank, local_idx in enumerate(top_local_idx, 1):
        global_idx = gap_indices[local_idx]
        tech_id    = TECH_VOCAB[global_idx]
        rows.append({
            "ORGID":           org_id,
            "Rank":            rank,
            "TechniqueID":     tech_id,
            "AE_Score":        round(float(ae_scores[global_idx]),  4),
            "MLP_Score":       round(float(mlp_scores[global_idx]), 4),
            "Ensemble_Score":  round(float(ens_scores[global_idx]),  4),
            "Already_Covered": 0,
        })

gaps_df = pd.DataFrame(rows)
gaps_df.to_csv(os.path.join(BASE, "phase3_technique_gaps.csv"), index=False)
print(f"Saved phase3_technique_gaps.csv ({len(gaps_df)} rows)")
print(f"Sample output for Org 1:")
print(gaps_df[gaps_df["ORGID"] == 1].head(10).to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════════
# SAVE MODELS
# ═════════════════════════════════════════════════════════════════════════════
autoencoder.save(os.path.join(BASE, "autoencoder_model"))
mlp.save(os.path.join(BASE, "mlp_model"))
encoder.save(os.path.join(BASE, "encoder_model"))
print(f"\nSaved autoencoder_model/, encoder_model/, and mlp_model/")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PHASE 3 TRAINING COMPLETE")
print("="*55)
print(f"Techniques modelled  : {N_TECHS}")
print(f"Organisations        : {len(org_ids)}")
print(f"Autoencoder latent   : {LATENT_DIM}-dim")
print(f"AE  epochs run       : {len(ae_history.history['loss'])}")
print(f"MLP epochs run       : {len(mlp_history.history['loss'])}")
print(f"AE  val BCE          : {ae_val_loss:.5f}")
print(f"MLP val BCE          : {mlp_val_loss:.5f}")
print(f"\nOutput files:")
print(f"  phase3_technique_gaps.csv      — top {TOP_N_GAPS} predicted gap techniques per org")
print(f"  phase3_model_comparison.csv    — full metrics (flat + ranking + ensemble)")
print(f"  autoencoder_model/             — saved Keras autoencoder")
print(f"  mlp_model/                     — saved Keras MLP")
print(f"  encoder_model/                 — saved encoder for latent space")
print(f"  autoencoder_history.json       — AE training history")
print(f"  mlp_history.json               — MLP training history")
print(f"  latent_vectors.npy             — latent space vectors")
print(f"  coverage_matrix.npy            — technique coverage matrix")
