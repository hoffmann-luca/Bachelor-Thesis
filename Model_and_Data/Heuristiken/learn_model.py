"""
Training von RandomForest-Modellen zur Algorithmuswahl (Meta-Solving).

Ablauf:
- lädt Feature-Tabelle train_data.csv
- trainiert pro Zeitlimit ein Modell + ein globales Modell
- nutzt Instance-basiertes Train/Test-Splitting
- speichert Modelle als .joblib
- schreibt alle Testvorhersagen nach model_test_results.csv
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from collections import Counter
import numpy as np, warnings

# Pfad zur Feature-/Label-Tabelle (pro Zeile: Instanz, Limit, Features, best_algo, ...)
DATA_PATH = Path(r"\features\v2\train_data.csv")
# Verzeichnis, in das trainierte Modelle und Testergebnisse geschrieben werden
MODEL_DIR = Path(r"\features\v3\models")
MODEL_DIR.mkdir(exist_ok=True)

# Sammeldatei mit allen Testvorhersagen (per-limit + global)
TEST_RESULTS_PATH = MODEL_DIR / "model_test_results.csv"

# feste Reihenfolge der Solver-Klassen (für Confusion-Matrix & Wahrscheinlichkeiten)
ALGOS = ["greedy", "grasp", "ils", "vns"]


def eval_and_print(y_true, y_pred, title=""):
    """
    Hilfsroutine: Accuracy + Confusion-Matrix + Klassifikationsreport ausgeben.
    Nutzt ALGOS als Klassenreihenfolge.
    """
    cm = confusion_matrix(y_true, y_pred, labels=ALGOS)
    acc = (y_true == y_pred).mean()
    print(f"{title} accuracy = {acc:.3f}")
    print("confusion (rows=true, cols=pred):")
    print("          " + "  ".join(f"{a:>6}" for a in ALGOS))
    for i, a in enumerate(ALGOS):
        row = "  ".join(f"{cm[i, j]:6d}" for j in range(len(ALGOS)))
        print(f"{a:>6}  {row}")
    print("\nreport:")
    print(classification_report(y_true, y_pred, labels=ALGOS, zero_division=0))


def split_by_instance(df, test_size=0.2, random_state=42):
    """
    Train/Test-Split auf Basis der instance_id:

    - es wird auf der Menge der Instanz-IDs gesplittet
    - alle Zeilen zu einer Instanz landen entweder komplett im Train- oder Test-Set
      (verhindert Data Leakage über gleiche Instanz bei unterschiedlichen Limits)
    """
    inst_ids = df["instance_id"].unique()
    train_ids, test_ids = train_test_split(
        inst_ids, test_size=test_size, random_state=random_state, shuffle=True
    )
    train_df = df[df["instance_id"].isin(train_ids)]
    test_df = df[df["instance_id"].isin(test_ids)]
    return train_df, test_df


def print_sbs(df: pd.DataFrame):
    """
    Gibt den Single-Best-Solver (SBS) global und pro Zeitlimit aus,
    anhand der Häufigkeit des empirisch besten Algorithmus (best_algo).
    """
    # global
    global_counts = df["best_algo"].value_counts()
    global_sbs_algo = global_counts.idxmax()
    global_sbs_acc = global_counts.max() / len(df)
    print(f"[SBS global] algo={global_sbs_algo} acc={global_sbs_acc:.3f} (n={len(df)})")

    # pro limit
    for L, sub in df.groupby("limit_s"):
        counts = sub["best_algo"].value_counts()
        sbs_algo = counts.idxmax()
        sbs_acc = counts.max() / len(sub)
        print(f"[SBS {L}s] algo={sbs_algo} acc={sbs_acc:.3f} (n={len(sub)})")


def eval_and_store(clf, X_test, y_test, test_df, tag, collector):
    """
    Wendet ein trainiertes Modell auf das Test-Set an, berechnet Vorhersagen
    + Klassenwahrscheinlichkeiten und sammelt alles in einem DataFrame.

    Parameters
    ----------
    clf        : trainierter Klassifikator
    X_test     : Test-Featurematrix
    y_test     : wahre Labels (best_algo)
    test_df    : Zeilen des vollständigen Original-DataFrames für das Test-Set
    tag        : Modellbezeichner (z.B. 'per-limit-20s' oder 'global')
    collector  : Liste, in die das Ergebnis-DataFrame angehängt wird
    """
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    out = pd.DataFrame({
        "instance_id": test_df["instance_id"].values,
        "limit_s": test_df["limit_s"].values,
        "true_algo": y_test.values,
        "pred_algo": y_pred,
        "correct": (y_pred == y_test.values).astype(int),
        "model_tag": tag,
    })

    # Sicherheitscheck: stimmen argmax(proba) und vorhergesagte Klasse überein?
    if not np.array_equal(y_pred, clf.classes_[proba.argmax(axis=1)]):
        warnings.warn("pred_algo ≠ argmax(proba_*) — Klassen/Spalten-Mapping prüfen.")

    # Wahrscheinlichkeiten in Spalten mit festen Namen schreiben (eine Spalte je Klasse)
    for i, cls in enumerate(clf.classes_):
        out[f"proba_{cls}"] = proba[:, i]

    collector.append(out)
    return y_pred


def train_per_limit(df: pd.DataFrame, result_collector: list):
    """
    Trainiert für jedes Zeitlimit (limit_s) ein separates RandomForest-Modell.

    Für jedes Limit:
      - Instanz-basierter Train/Test-Split
      - Featureselektion durch Weglassen von ID- und Zielspalten
      - einfache Klassen-Gewichtung über inverse Häufigkeiten + manuelle Tweaks
      - Training + Evaluation
      - Speichern von Modell + Testvorhersagen
    """
    limits = sorted(df["limit_s"].unique())
    for L in limits:
        sub = df[df["limit_s"] == L].copy()
        if len(sub) < 30:
            print(f"[warn] zu wenig Daten für {L}s (n={len(sub)}) -> skip")
            continue

        train_df, test_df = split_by_instance(sub)

        # Sanity-Check: keine Instanz darf sowohl in Train als auch Test vorkommen
        train_ids = set(train_df["instance_id"])
        test_ids = set(test_df["instance_id"])
        inter = train_ids & test_ids
        print("gemeinsame IDs:", inter)
        assert len(inter) == 0, "Train/Test NICHT disjunkt!"

        # Feature-Spalten definieren (IDs, Ziel und Aggregatspalten werden ausgeschlossen)
        drop_cols = {
            "instance_id",
            "limit_s",
            "best_algo",
            "mean_score",
            "mean_remaining",
            "mean_used_time_s",
            "n_runs",
        }
        feat_cols = [c for c in sub.columns if c not in drop_cols]

        Xtr = train_df[feat_cols]
        ytr = train_df["best_algo"]
        Xte = test_df[feat_cols]
        yte = test_df["best_algo"]

        # 1) inverse Häufigkeiten als Basis für Klassen-Gewichte
        freq = Counter(ytr)
        inv = {k: 1.0 / v for k, v in freq.items()}
        # 2) normieren, damit Summe der Gewichte ~ 1 ist
        s = sum(inv.values())
        class_w = {k: (v / s) for k, v in inv.items()}
        # 3) optionale Feinjustierung einzelner Klassen
        class_w["grasp"] *= 0.7  # GRASP leicht runtergewichten
        class_w["vns"] *= 1.2    # VNS leicht hochgewichten

        # sample weights je Trainingszeile aus Klassen-Gewichten ableiten
        w = np.array([class_w[c] for c in ytr])

        n_estimators = [500]
        for n_esti in n_estimators:
            clf = RandomForestClassifier(
                n_estimators=n_esti,
                random_state=42,
                # class_weight="balanced",
                n_jobs=-1,
            )
            print(f"[estimator] {n_esti} for [limit] {L}s")
            # Hinweis: 'if L == 20 or 40' ist in Python immer True, d.h. sample_weight wird
            # effektiv für alle Limits verwendet (logische Besonderheit, kein Funktionsfehler).
            if L == 20 or 40:
                clf.fit(Xtr, ytr, sample_weight=w)
            else:
                clf.fit(Xtr, ytr)

            # Trainingsevaluation (Overfitting-Check)
            eval_and_print(ytr.values, clf.predict(Xtr), title=f"[train] {L}s")

            # Testevaluation + Speicherung der Vorhersagen
            ypred = eval_and_store(
                clf, Xte, yte, test_df, f"per-limit-{L}s", result_collector
            )
            eval_and_print(yte.values, ypred, title=f"[test] {L}s")

            acc = accuracy_score(yte, ypred)
            print(f"[per-limit] {L}s: acc={acc:.3f}  (n={len(sub)})")
            print("-------------------------------------------------------")

        # pro-Limit-Modell mit Featureliste abspeichern
        joblib.dump(
            {"model": clf, "features": feat_cols, "limit_s": L},
            MODEL_DIR / f"rf_per_limit_{int(L)}s.joblib",
        )


def train_global(df: pd.DataFrame, result_collector: list):
    """
    Trainiert ein globales RandomForest-Modell über alle Limits.

    Unterschied zu train_per_limit:
      - limit_s bleibt hier Feature und wird nicht gedroppt
      - ein gemeinsames Modell entscheidet für alle Limit-Werte
    """
    train_df, test_df = split_by_instance(df)

    # sicherstellen, dass Instanzen nicht in beiden Splits vorkommen
    train_ids = set(train_df["instance_id"])
    test_ids = set(test_df["instance_id"])
    inter = train_ids & test_ids
    print("gemeinsame IDs:", inter)
    assert len(inter) == 0, "Train/Test NICHT disjunkt!"

    drop_cols = {
        "instance_id",
        "best_algo",
        "mean_score",
        "mean_remaining",
        "mean_used_time_s",
        "n_runs",
    }
    feat_cols = [c for c in df.columns if c not in drop_cols]

    Xtr = train_df[feat_cols]
    ytr = train_df["best_algo"]
    Xte = test_df[feat_cols]
    yte = test_df["best_algo"]

    # Klassen-Gewichte wie im per-Limit-Fall
    freq = Counter(ytr)
    inv = {k: 1.0 / v for k, v in freq.items()}
    s = sum(inv.values())
    class_w = {k: (v / s) for k, v in inv.items()}
    class_w["grasp"] *= 0.7
    class_w["vns"] *= 1.2

    w = np.array([class_w[c] for c in ytr])

    n_estimators = [500]
    for n_esti in n_estimators:
        clf = RandomForestClassifier(
            n_estimators=n_esti,
            random_state=42,
            # class_weight="balanced",
            n_jobs=-1,
        )
        print(f"[estimator] {n_esti} for [global]")
        clf.fit(Xtr, ytr, sample_weight=w)

        ypred = eval_and_store(clf, Xte, yte, test_df, "global", result_collector)
        eval_and_print(yte.values, ypred, title="[test] global")

        acc = accuracy_score(yte, ypred)
        print(f"[global] acc={acc:.3f}  (n={len(df)})")
        print("-------------------------------------------------------")

    # globales Modell abspeichern
    joblib.dump(
        {"model": clf, "features": feat_cols},
        MODEL_DIR / "rf_global.joblib",
    )


def main():
    """
    Oberste Steuerfunktion:

    - lädt Trainingsdaten
    - gibt SBS-Übersicht aus
    - trainiert per-Limit-Modelle und globales Modell
    - schreibt alle Testvorhersagen in eine gemeinsame CSV
    """
    df = pd.read_csv(DATA_PATH)
    print_sbs(df)

    all_results = []  # hier sammeln wir alle Test-Vorhersagen

    train_per_limit(df, all_results)
    train_global(df, all_results)

    # alle Test-Ergebnisse zusammenführen und auf Platte schreiben
    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(TEST_RESULTS_PATH, index=False)
        print(f"[info] Test-Ergebnisse gespeichert unter: {TEST_RESULTS_PATH}")


if __name__ == "__main__":
    main()
