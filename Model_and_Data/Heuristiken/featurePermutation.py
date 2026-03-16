"""
Feature-Permutationsanalyse für RandomForest-Klassifikator.

Pipeline:
- Lädt Trainingsdaten (Features + best_algo als Zielvariable)
- Trennt globales Modell und pro-Zeitlimit-Modelle
- Instanzweise Train/Eval-Split
- Feature-Selektion (Zero-Variance + grobe Korrelationspruning)
- Trainiert RandomForest
- Berechnet Permutation Importance (Accuracy + macro-F1)
- Berechnet Per-Klasse-Permutation Importance (ΔF1 je Klasse)
- Kleine Ablation: Features mit nicht-positiver PI werden entfernt und Modell erneut evaluiert
- Schreibt Berichte nach OUTDIR
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.inspection import permutation_importance

# ----------------- Pfade anpassen -----------------
DATA = Path(r"\features\v2\train_data.csv")                 # Trainingsdaten mit Features + best_algo
OUTDIR = Path(r"\results\v4\feature_audit_out"); OUTDIR.mkdir(exist_ok=True, parents=True)

TARGET_COL = "best_algo"  # Zielvariable: empirisch bester Algorithmus
ID_COLS = ["instance_id","best_algo","mean_score","mean_remaining","mean_used_time_s","n_runs"]                     # Metadaten, die nicht als Features verwendet werden
GLOBAL_FEATURE_LIMIT_COL = "limit_s"          # globales Modell: limit_s wird als zusätzliche Feature-Spalte verwendet
# ---------------------------------------------------

RANDOM_STATE = 42

def _is_feature(col: str) -> bool:
    """
    Entscheidet, ob eine Spalte als Feature verwendet wird.

    Ausschlüsse:
    - ID_COLS (Instanz-/Run-Metadaten)
    - TARGET_COL (Zielvariable)

    Alles andere wird als Feature interpretiert, inkl. 'limit_s' für das globale Modell.
    """
    if col in ID_COLS: return False
    if col == TARGET_COL: return False
    # Alles andere sind Features (inkl. global 'limit_s')
    return True

def split_by_limit(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Erzeugt Teil-Datenframes für:
    - 'global': alle Limits zusammen
    - je Limit_s einen separaten Teil-Frame (Key = str(limit_s))

    So kann man ein globales Modell und je Limit ein eigenes Modell trainieren.
    """
    parts = {"global": df.copy()}
    for L, sub in df.groupby("limit_s"):
        parts[str(L)] = sub.copy()
    return parts

def make_train_eval_split(df: pd.DataFrame, eval_frac=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Erzeugt einen instanzweisen Train/Eval-Split.

    Wichtig: Auf Ebene der instance_id splitten, damit Instanzen nicht gleichzeitig
    im Training und im Eval-Set vorkommen (Vermeidung von Data Leakage).
    """
    ids = df["instance_id"].unique()
    print(f"features={len(df.columns)}")
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(ids)
    cut = int(len(ids) * (1 - eval_frac))
    train_ids = set(ids[:cut])
    eval_ids  = set(ids[cut:])
    train = df[df["instance_id"].isin(train_ids)].copy()
    eval_  = df[df["instance_id"].isin(eval_ids)].copy()
    return train, eval_

def zero_variance_filter(X: pd.DataFrame, cols: List[str]) -> List[str]:
    """
    Entfernt Features mit Varianz 0 (konstante Spalten).
    Gibt (keep_cols, dropped_cols) zurück.
    """
    var = X[cols].var(numeric_only=True)
    keep = [c for c in cols if var.get(c, 1.0) > 0.0]
    dropped = sorted(set(cols) - set(keep))
    return keep, dropped

def correlation_prune(X: pd.DataFrame, cols: List[str], thr=0.98) -> List[str]:
    """
    Grobes Korrelations-Pruning:
    - berechnet |corr| zwischen allen Featurepaaren
    - wählt iterativ ein "repräsentatives" Feature pro Korrelationscluster (> thr)
    - reduziert damit stark korrelierte Duplikate

    Gibt (keep_cols, pruned_cols) zurück.
    """
    keep = []
    dropped = set()
    # robust bei wenigen Spalten
    C = X[cols].corr(numeric_only=True).fillna(0.0).abs()
    used = set()
    for c in cols:
        if c in used:
            continue
        # markiere hochkorrelierte Partner
        partners = [c2 for c2 in cols if c2 != c and C.loc[c, c2] > thr]
        used.add(c)
        used.update(partners)
        keep.append(c)
    pruned = sorted(set(cols) - set(keep))
    return keep, pruned

def eval_model(tag: str, train: pd.DataFrame, eval_: pd.DataFrame, feature_cols: List[str]):
    """
    Trainiert ein RandomForest-Modell und wertet es aus:

    - Training auf 'train' mit den übergebenen feature_cols.
    - Standard-Metriken: Accuracy und macro-F1 auf dem Eval-Set.
    - Permutation Importance für Accuracy und macro-F1.
    - Per-Klasse-Permutation Importance: ΔF1 nur für eine Klasse.
    - Schreibt:
        * perm_importance_{tag}.csv
        * perm_importance_per_class_{tag}.csv
        * summary_{tag}.json

    Zusätzlich:
    - kleine Ablation: alle Features mit pi_f1_macro_mean <= 0 werden entfernt,
      Modell wird neu trainiert und erneut evaluiert.
    """
    X_tr, y_tr = train[feature_cols], train[TARGET_COL]
    X_ev, y_ev = eval_[feature_cols], eval_[TARGET_COL]

    clf = RandomForestClassifier(
        n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_ev)
    acc = accuracy_score(y_ev, y_pred)
    f1_macro = f1_score(y_ev, y_pred, average="macro")

    # Permutation Importance: global für Accuracy und macro-F1
    perm_acc = permutation_importance(clf, X_ev, y_ev, n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1, scoring="accuracy")
    perm_f1  = permutation_importance(clf, X_ev, y_ev, n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1, scoring="f1_macro")

    # Per-Klasse-Permutation (ΔF1 dieser Klasse):
    # definiert ein Scoring, welches nur den F1 für eine Klasse betrachtet.
    classes = np.unique(y_ev)
    per_class_rows = []
    for cls in classes:
        scoring = (lambda est, X, y: f1_score(y, est.predict(X), labels=[cls], average="macro", zero_division=0))
        perm_cls = permutation_importance(clf, X_ev, y_ev, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1, scoring=scoring)
        for name, imp, std in zip(feature_cols, perm_cls.importances_mean, perm_cls.importances_std):
            per_class_rows.append({
                "scope": tag, "feature": name, "class": cls,
                "pi_f1cls_mean": float(imp), "pi_f1cls_std": float(std)
            })
    per_class_df = pd.DataFrame(per_class_rows)

    # Aggregation der globalen Permutation-Importance-Ergebnisse
    report = pd.DataFrame({
        "feature": feature_cols,
        "pi_acc_mean": perm_acc.importances_mean,
        "pi_acc_std":  perm_acc.importances_std,
        "pi_f1_macro_mean": perm_f1.importances_mean,
        "pi_f1_macro_std":  perm_f1.importances_std,
    }).sort_values("pi_f1_macro_mean", ascending=False)

    OUTDIR.mkdir(exist_ok=True, parents=True)
    report.to_csv(OUTDIR / f"perm_importance_{tag}.csv", index=False)
    per_class_df.to_csv(OUTDIR / f"perm_importance_per_class_{tag}.csv", index=False)

    # kleine Ablation: Features mit (pi_f1_macro_mean <= 0) droppen und re-evaluieren
    low_feats = set(report[report["pi_f1_macro_mean"] <= 0.0]["feature"])
    if low_feats:
        keep_feats = [c for c in feature_cols if c not in low_feats]
        clf2 = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
        clf2.fit(X_tr[keep_feats], y_tr)
        y2 = clf2.predict(X_ev[keep_feats])
        acc2 = accuracy_score(y_ev, y2)
        f1m2 = f1_score(y_ev, y2, average="macro")
    else:
        keep_feats, acc2, f1m2 = feature_cols, acc, f1_macro

    summary = {
        "scope": tag,
        "n_train": int(len(train)), "n_eval": int(len(eval_)),
        "n_features": len(feature_cols),
        "acc": round(acc, 4), "f1_macro": round(f1_macro, 4),
        "acc_after_drop_nonpos": round(acc2, 4),
        "f1_after_drop_nonpos": round(f1m2, 4),
        "dropped_nonpos_features": sorted(list(low_feats)),
    }
    with open(OUTDIR / f"summary_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # kurze Konsolenübersicht: Gesamtleistung und Top/Bottom-Features
    print(f"\n[{tag}] acc={acc:.3f}, f1_macro={f1_macro:.3f}, features={len(feature_cols)}")
    print("Top-10 by PI (macro-F1):")
    print(report.head(10).to_string(index=False))
    print("\nBottom-10 by PI (macro-F1):")
    print(report.tail(10).to_string(index=False))

def main():
    """
    Gesamtsteuerung der Feature-Permutationsanalyse:

    - lädt Trainingsdaten aus DATA
    - erzeugt globale und per-Limit-Subsets
    - führt für jeden 'scope' (global + limit_s) eine eigene
      Train/Eval-Split-Phase, Feature-Selektion und Modell-Evaluation durch.
    """
    df = pd.read_csv(DATA)
    assert TARGET_COL in df.columns, f"{TARGET_COL} fehlt."
    parts = split_by_limit(df)

    for tag, frame in parts.items():
        # instanzweiser Train/Eval-Split
        train, eval_ = make_train_eval_split(frame, eval_frac=0.2)

        # Featureliste generieren
        feature_cols = [c for c in frame.columns if _is_feature(c)]
        # in den per-Limit-Modellen 'limit_s' als Feature ausschließen
        if tag != "global":
            feature_cols = [c for c in feature_cols if c != GLOBAL_FEATURE_LIMIT_COL]
        # Sanity-Filter: konstante Features entfernen
        feature_cols, dropped_zero = zero_variance_filter(train, feature_cols)
        # Redundanz grob eindampfen (optional: niedrigeres thr macht aggressiver)
        feature_cols, dropped_corr = correlation_prune(train, feature_cols, thr=0.98)

        # Protokoll der entfernten Features
        pd.Series(sorted(dropped_zero)).to_csv(OUTDIR / f"dropped_zero_variance_{tag}.csv", index=False, header=False)
        pd.Series(sorted(dropped_corr)).to_csv(OUTDIR / f"dropped_highcorr_{tag}.csv", index=False, header=False)

        # Modell trainieren und Permutation Importance berechnen
        eval_model(tag, train, eval_, feature_cols)

if __name__ == "__main__":
    main()
