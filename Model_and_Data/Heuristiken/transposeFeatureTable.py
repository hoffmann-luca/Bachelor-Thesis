"""
Transpose/Weit-Format-Erzeugung für Graph-Features

Idee:
- Eingabe: features_all.parquet im "long" Format
  (mehrere Zeilen pro Instanz, je nach builder/radius/k usw.)
- Ausgabe: features_all_wide.csv im "wide" Format
  (genau eine Zeile pro instance_id mit allen Feature-Spalten)

Schritte:
  1. Einlesen & Spalten vereinheitlichen
  2. Aufräumen (überflüssige Spalten, Radius-String -> float)
  3. Basis-Tabelle mit globalen Features pro Instanz
  4. Für jeden Graph-Builder (radius, knn, nng, delaunay, mst)
     einen eigenen Feature-Block bilden und per instance_id joinen
"""

import pandas as pd
from pathlib import Path

# Eingabe: long-Format aller Graph-Features
IN_PATH = Path(r"\features\v2\features_all.parquet")
# Ausgabe: wide-Format mit einer Zeile pro Instanz
OUT_PATH = Path(r"\features\v2\features_all_wide.csv")


# 1. Einlesen der long-Tabelle
df = pd.read_parquet(IN_PATH)

# 2. Instanz-ID-Spalte vereinheitlichen
#    Falls nur dataset_id existiert, in instance_id umbenennen
if "instance_id" not in df.columns and "dataset_id" in df.columns:
    df = df.rename(columns={"dataset_id": "instance_id"})

# 3. Offensichtlich überflüssige / technische Spalten entfernen
for col in ["source_path", "qhull_options"]:
    if col in df.columns:
        df = df.drop(columns=col)

# 4. radius säubern:
#    - teilweise kommt radius als String mit Komma ("0,5")
#    - erst "," -> ".", dann in float konvertieren
if "radius" in df.columns:
    df["radius"] = (
        df["radius"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df["radius"] = pd.to_numeric(df["radius"], errors="coerce")

# ---------------------------------------------------------------------
# BASIS: globale Spalten, die für alle Builder identisch sind
#        (pro Instanz nur eine Zeile)
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
base = (
    df[["instance_id"] + [c for c in GLOBAL_COLS if c in df.columns]]
    .drop_duplicates("instance_id")
    .set_index("instance_id")
)

# ---------------------------------------------------------------------
# HILFSFUNKTIONEN

def merge_builder(final_df: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """
    Joint einen Builder-spezifischen Block (part) anhand von instance_id
    in die bereits vorhandene Basistabelle (final_df).
    """
    return final_df.join(part, how="left")


def make_suffix_from_radius(r: float) -> str:
    """
    Bildet aus einem Radius-Wert ein Suffix für Spaltennamen.
    Beispiel: 0.5 -> "0_5", 2.0 -> "2_0"
    """
    return str(r).replace(".", "_")


# ---------------------------------------------------------------------
# 1) build_radius_graph  (nur Basis-Graph-Features, je Radius ein Block)

RADIUS_BASE_FEATS = [
    "degree_min",
    "degree_max",
    "degree_avg",
    "components",
    "component_largest",
    "component_avg_size",
    "clustering_avg",
]

rad = df[df["builder"] == "build_radius_graph"].copy()
radius_values = sorted(rad["radius"].dropna().unique())

rad_parts = []
for r in radius_values:
    # Teil-DF für einen bestimmten Radius
    sub = rad[rad["radius"] == r].copy()
    keep = ["instance_id"] + [f for f in RADIUS_BASE_FEATS if f in sub.columns]
    sub = sub[keep].drop_duplicates("instance_id")
    sub = sub.set_index("instance_id")
    # Spalten prefixen: rad_<r>__*
    suf = make_suffix_from_radius(r)
    sub = sub.add_prefix(f"rad_{suf}__")
    rad_parts.append(sub)

# alle Radius-Blöcke spaltenweise aneinanderhängen und in base injoinen
if rad_parts:
    rad_wide = pd.concat(rad_parts, axis=1)
    base = merge_builder(base, rad_wide)

# ---------------------------------------------------------------------
# 2) build_knn_graph  (gleiche Basis-Features, je k ein Block)

KNN_BASE_FEATS = RADIUS_BASE_FEATS  # identisches Featureset

knn = df[df["builder"] == "build_knn_graph"].copy()
knn_values = sorted(knn["k"].dropna().unique())

knn_parts = []
for k in knn_values:
    # Teil-DF für einen konkreten k-Wert
    sub = knn[knn["k"] == k].copy()
    keep = ["instance_id"] + [f for f in KNN_BASE_FEATS if f in sub.columns]
    sub = sub[keep].drop_duplicates("instance_id").set_index("instance_id")
    # Spalten prefixen: knn_<k>__*
    sub = sub.add_prefix(f"knn_{int(k)}__")
    knn_parts.append(sub)

if knn_parts:
    knn_wide = pd.concat(knn_parts, axis=1)
    base = merge_builder(base, knn_wide)

# ---------------------------------------------------------------------
# 3) build_nng_via_delaunay  (Nearest-Neighbor-Graph: mutual / plain)

NNG_BASE_FEATS = RADIUS_BASE_FEATS

nng = df[df["builder"] == "build_nng_via_delaunay"].copy()
# mutual kann fehlen -> als False (plain) interpretieren
if "mutual" not in nng.columns:
    nng["mutual"] = False

nng_parts = []
for flag, sub in nng.groupby(nng["mutual"].astype(bool)):
    # label: "mut" für mutual=True, sonst "plain"
    label = "mut" if flag else "plain"
    keep = ["instance_id"] + [f for f in NNG_BASE_FEATS if f in sub.columns]
    sub = sub[keep].drop_duplicates("instance_id").set_index("instance_id")
    sub = sub.add_prefix(f"nng_{label}__")
    nng_parts.append(sub)

if nng_parts:
    nng_wide = pd.concat(nng_parts, axis=1)
    base = merge_builder(base, nng_wide)

# ---------------------------------------------------------------------
# 4) build_delaunay_graph  (eine Zeile pro Instanz, Basis-Features)

delau = df[df["builder"] == "build_delaunay_graph"].copy()
if not delau.empty:
    keep = ["instance_id"] + [f for f in RADIUS_BASE_FEATS if f in delau.columns]
    delau = delau[keep].drop_duplicates("instance_id").set_index("instance_id")
    delau = delau.add_prefix("delaunay__")
    base = merge_builder(base, delau)

# ---------------------------------------------------------------------
# 5) build_mst_via_delaunay  (Basis- + MST-spezifische Features)

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
mst = df[df["builder"] == "build_mst_via_delaunay"].copy()
if not mst.empty:
    keep = ["instance_id"]
    keep += [f for f in RADIUS_BASE_FEATS if f in mst.columns]
    keep += [f for f in MST_FEATS if f in mst.columns]
    mst = mst[keep].drop_duplicates("instance_id").set_index("instance_id")
    mst = mst.add_prefix("mst__")
    base = merge_builder(base, mst)

# ---------------------------------------------------------------------
# finale Wide-Tabelle schreiben (eine Zeile pro instance_id)
final = base.reset_index()
final.to_parquet(OUT_PATH, index=False)
# final.to_csv(r"...\features_all_wide.parquet", index=False)
print("fertig:", OUT_PATH)
