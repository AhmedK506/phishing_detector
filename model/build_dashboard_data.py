import json

# ── All dashboard stats in one place ──────────────────────
# These come from your actual runs. Edit any that differ.
data = {
    # Preprocessing (from preprocess.py output)
    "preprocessing": {
        "raw_rows": 651191,
        "malformed_removed": 61,
        "duplicates_removed": 10071,
        "cleaned_rows": 641059,
        "missing_values": 0
    },
    # Dataset composition (after adding Tranco)
    "dataset": {
        "kaggle_rows": 641059,
        "tranco_added": 30000,
        "total_training": 671059,
        "safe_pct": 66.78,
        "malicious_pct": 33.22,
        "imbalance": "2.01 : 1",
        "num_features": 35
    },
    # Model performance (from your training evaluation)
    "metrics": {
        "accuracy": 97.01,
        "precision": 90.75,
        "recall": 92.36,
        "f1": 91.55
    },
    # Confusion matrix (from confusion_matrix.py — the combined model)
    "confusion": {
        "tn": 108462, "fp": 2215,
        "fn": 1797, "tp": 21738
    },
    # Before/after held-out evaluation (from evaluate.py)
    "evaluation": {
        "old_accuracy": 47.0, "new_accuracy": 99.0,
        "old_fp": 50, "new_fp": 0,
        "old_recall": 94.0, "new_recall": 98.0
    },
    # Model info
    "model": {
        "algorithm": "Random Forest",
        "type": "Supervised · Binary Classification",
        "tuning": "RandomizedSearchCV",
        "balancing": "class_weight='balanced'"
    }
}

with open('../data/dashboard_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Saved dashboard_data.json")
print(json.dumps(data, indent=2))