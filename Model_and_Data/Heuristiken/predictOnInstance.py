# predict_new_instance.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib

# Feature-Extraktion für eine einzelne Instanz (kommt aus deinem Batch-Extractor)
from batch_PropertiesExtractor import one_file

# ------------------------------------------------------------
# 1) long -> wide (1 Zeile pro instance_id)
#    (entspricht der Logik aus changeParquetTable.py)
# ------------------------------------------------------------

# globale Datensatz-Features (nicht graph-spezifisch)
GLOBAL_COLS = [
    "budget",
    "nodes",
    "hull_vertices",
    "hull_area",
    "hull_perimeter",
    "hull_compactness",
    "density",
    "budget_per_node",
]

# Basis-Features, die bei Radius-/kNN-/Delaunay-/NNG-Graphen verwendet werden
RADIUS_BASE_FEATS = [
    "degree_min",
    "degree_max",
    "degree_avg",
    "components",
    "component_largest",
    "component_avg_size",
    "clustering_avg",
]

# zusätzliche Features, die nur für den MST relevant sind
MST_FEATS = [
    "mst_edge_sum",
    "mst_edge_max",
    "mst_leaves",
    "mst_diameter",
    "mst_edge_sum_per_sqrtA",
    "mst_edge_max_per_sqrtA",
    "mst_edge_sum_units_s",
    "mst_edge_sum_units_m",
]


def _merge_builder(base: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """
    Hilfsfunktion: mergen eines Builder-spezifischen Wide-Blocks (part)
    in die Basis-Tabelle (base) per Index (instance_id).
    """
    return base.join(part, how="left")


def _make_suffix_from_radius(r: float) -> str:
    """Radiuswert → String-Suffix, z.B. 0.5 → '0_5'."""
    return str(r).replace(".", "_")


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformiert die "long"-Tabelle (mehrere Zeilen pro Instanz/Builder)
    in eine "wide"-Tabelle (eine Zeile pro Instanz mit allen Feature-Spalten).

    Erwartung:
    - df enthält u.a. Spalten:
      * instance_id oder dataset_id
      * builder
      * radius bzw. k (je nach Builder)
      * globale Stats + Graph-Stats (degree_*, components, clustering_avg, ...)
    """
    # evtl. dataset_id -> instance_id umbenennen
    if "instance_id" not in df.columns and "dataset_id" in df.columns:
        df = df.rename(columns={"dataset_id": "instance_id"})

    # Basisblock: globale Features pro Instanz
    base = (
        df[["instance_id"] + [c for c in GLOBAL_COLS if c in df.columns]]
        .drop_duplicates("instance_id")
        .set_index("instance_id")
    )

    # 1) Radius-Graphen: für jeden Radius ein eigener Block mit Prefix rad_<r>__
    rad = df[df["builder"] == "build_radius_graph"].copy()
    if not rad.empty:
        radius_values = sorted(rad["radius"].dropna().unique())
        parts = []
        for r in radius_values:
            sub = rad[rad["radius"] == r].copy()
            keep = ["instance_id"] + [f for f in RADIUS_BASE_FEATS if f in sub.columns]
            sub = sub[keep].drop_duplicates("instance_id").set_index("instance_id")
            suf = _make_suffix_from_radius(r)
            sub = sub.add_prefix(f"rad_{suf}__")
            parts.append(sub)
        if parts:
            rad_wide = pd.concat(parts, axis=1)
            base = _merge_builder(base, rad_wide)

    # 2) kNN-Graphen: je k ein Block mit Prefix knn_<k>__
    knn = df[df["builder"] == "build_knn_graph"].copy()
    if not knn.empty:
        k_values = sorted(knn["k"].dropna().unique())
        parts = []
        for k in k_values:
            sub = knn[knn["k"] == k].copy()
            keep = ["instance_id"] + [f for f in RADIUS_BASE_FEATS if f in sub.columns]
            sub = sub[keep].drop_duplicates("instance_id").set_index("instance_id")
            sub = sub.add_prefix(f"knn_{int(k)}__")
            parts.append(sub)
        if parts:
            knn_wide = pd.concat(parts, axis=1)
            base = _merge_builder(base, knn_wide)

    # 3) NNG-Graphen: plain vs. mutual → nng_plain__* und nng_mut__*
    nng = df[df["builder"] == "build_nng_via_delaunay"].copy()
    if not nng.empty:
        if "mutual" not in nng.columns:
            nng["mutual"] = False
        parts = []
        for flag, sub in nng.groupby(nng["mutual"].astype(bool)):
            label = "mut" if flag else "plain"
            keep = ["instance_id"] + [f for f in RADIUS_BASE_FEATS if f in sub.columns]
            sub = sub[keep].drop_duplicates("instance_id").set_index("instance_id")
            sub = sub.add_prefix(f"nng_{label}__")
            parts.append(sub)
        if parts:
            nng_wide = pd.concat(parts, axis=1)
            base = _merge_builder(base, nng_wide)

    # 4) Delaunay-Graphen: ein Block mit Prefix delaunay__*
    delau = df[df["builder"] == "build_delaunay_graph"].copy()
    if not delau.empty:
        keep = ["instance_id"] + [f for f in RADIUS_BASE_FEATS if f in delau.columns]
        delau = delau[keep].drop_duplicates("instance_id").set_index("instance_id")
        delau = delau.add_prefix("delaunay__")
        base = _merge_builder(base, delau)

    # 5) MST-Graphen: Basis- + MST-spezifische Features, Prefix mst__*
    mst = df[df["builder"] == "build_mst_via_delaunay"].copy()
    if not mst.empty:
        keep = ["instance_id"]
        keep += [f for f in RADIUS_BASE_FEATS if f in mst.columns]
        keep += [f for f in MST_FEATS if f in mst.columns]
        mst = mst[keep].drop_duplicates("instance_id").set_index("instance_id")
        mst = mst.add_prefix("mst__")
        base = _merge_builder(base, mst)

    return base.reset_index()


# ------------------------------------------------------------
# 2) Feature-Auswahl exakt wie FeatureVectorkeeper.py
# ------------------------------------------------------------

# Kernfeatures (nicht graph-spezifisch)
KEEP_CORE = [
    "budget",
    "nodes",
    "hull_vertices",
    "hull_area",
    "hull_perimeter",
    "hull_compactness",
    "density",
    "budget_per_node",
]

# alle Builder-/Graph-Features, die in der BA als relevant ausgewählt wurden
KEEP_BUILDER = [
    # radius
    "rad_0_5__degree_min",
    "rad_0_5__degree_max",
    "rad_0_5__degree_avg",
    "rad_0_5__component_largest",
    "rad_0_5__clustering_avg",
    "rad_1_0__degree_min",
    "rad_1_0__clustering_avg",

    # knn_3
    "knn_3__degree_max",
    "knn_3__degree_avg",
    "knn_3__components",
    "knn_3__component_largest",
    "knn_3__component_avg_size",
    "knn_3__clustering_avg",

    # knn_5
    "knn_5__degree_max",
    "knn_5__degree_avg",
    "knn_5__components",
    "knn_5__component_largest",
    "knn_5__component_avg_size",
    "knn_5__clustering_avg",

    # knn_8
    "knn_8__degree_max",
    "knn_8__degree_avg",
    "knn_8__components",
    "knn_8__component_largest",
    "knn_8__component_avg_size",
    "knn_8__clustering_avg",

    # knn_12
    "knn_12__degree_max",
    "knn_12__degree_avg",
    "knn_12__components",
    "knn_12__component_largest",
    "knn_12__component_avg_size",
    "knn_12__clustering_avg",

    # nng_plain
    "nng_plain__degree_max",
    "nng_plain__degree_avg",
    "nng_plain__components",
    "nng_plain__component_largest",
    "nng_plain__component_avg_size",
    "nng_plain__clustering_avg",

    # delaunay
    "delaunay__degree_max",
    "delaunay__degree_avg",
    "delaunay__clustering_avg",
]

# MST-Featureauswahl
KEEP_MST = [
    "mst__degree_avg",
    "mst__mst_edge_sum",
    "mst__mst_edge_max",
    "mst__mst_leaves",
    "mst__mst_diameter",
    "mst__mst_edge_sum_per_sqrtA",
    "mst__mst_edge_max_per_sqrtA",
    "mst__mst_edge_sum_units_s",
    "mst__mst_edge_sum_units_m",
]

# Gesamt-Featureliste (Reihenfolge ist nicht kritisch)
KEEP_COLS = KEEP_CORE + KEEP_BUILDER + KEEP_MST


def select_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Wählt aus der Wide-Tabelle genau die Spalten aus, die in KEEP_COLS stehen
    und tatsächlich vorhanden sind. Liefert nur die Feature-Matrix zurück.
    """
    cols = [c for c in KEEP_COLS if c in df_wide.columns]
    out = df_wide[cols].copy()
    return out


# ------------------------------------------------------------
# 3) Modell laden und vorhersagen
# ------------------------------------------------------------

def load_model(model_dir: str, mode: str, limit_s: float | None = None):
    """
    Lädt ein gespeichertes RandomForest-Modell:

    mode = 'global'    → rf_global.joblib
    mode = 'per-limit' → rf_per_limit_<limit_s>s.joblib
    """
    model_dir = Path(model_dir)
    if mode == "global":
        path = model_dir / "rf_global.joblib"
    else:  # per-limit
        path = model_dir / f"rf_per_limit_{int(limit_s)}s.joblib"
    return joblib.load(path)


def predict_on_file(dt: pd.DataFrame, model_dir: str, mode: str, limit_s: float):
    """
    Wendet ein geladenes Modell auf EINEN Featurevektor an.

    Wichtig:
    - dt ist hier die Feature-Tabelle (eine Zeile) für die Instanz.
    - Das Modell erwartet die Spaltenreihenfolge 'bundle["features"]'.

    Hinweis:
    - Diese Implementierung nutzt intern die globale Variable df_feat
      (dt wird aktuell nicht verwendet). Das spiegelt den Originalcode wider.
    """
    bundle = load_model(model_dir, mode, limit_s)
    clf = bundle["model"]
    feat_order = bundle["features"]

    # WICHTIG: beim GLOBALEN Modell limit_s als Feature setzen
    if mode == "global":
        df_feat["limit_s"] = float(limit_s)
        df_feat.to_csv("newFeatureVec_time.csv", index=False)

    # Feature-Matrix in der Reihenfolge des Trainingsmodells aufbauen
    X = df_feat.reindex(columns=feat_order, fill_value=0.0)
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]

    return pred, dict(zip(clf.classes_, proba))


if __name__ == "__main__":
    # Beispielpfade (eine Instanzdatei + Modellverzeichnis)
    TXT = r"\Test\test\instance.txt"
    MODELS = r"\features\v2\models"

    # 1) Feature-Extraktion:
    #    - one_file: Graph-Features im "long"-Format
    #    - long_to_wide: eine Zeile pro Instanz
    #    - select_features: Reduktion auf die genutzten Feature-Spalten
    df_long = one_file(TXT)
    df_wide = long_to_wide(df_long)
    df_feat = select_features(df_wide)
    df_feat.to_csv("newFeatureVec.csv", index=False)

    # 2) Vorhersage mit dem globalen Modell für verschiedene Zeitlimits
    for limit in [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]:
        algo, probs = predict_on_file(df_feat, MODELS, mode="global", limit_s=limit)
        print(f"global {limit}s:", algo, probs)

    print("------------------------------------------------------------------------------")

    # 3) Vorhersage mit per-Limit-Modellen (falls trainiert und vorhanden)
    for limit in [1.0, 5.0, 10.0, 20.0, 40.0]:
        algo, probs = predict_on_file(df_feat, MODELS, mode="per-limit", limit_s=limit)
        print(f"per-limit {limit}s:", algo, probs)
