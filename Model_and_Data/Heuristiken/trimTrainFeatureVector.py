"""
Reduziert die breite Feature-Tabelle auf einen kleineren, kuratierten Satz von Features.

Ablauf:
1. features_all_wide.csv einlesen (eine Zeile pro Instanz, viele Feature-Spalten)
2. Kern- und ausgewählte Graph-Features definieren (keep_*-Listen)
3. Tabelle auf diese Spalten beschränken
4. Ergebnis als features_all_wide_trim.csv speichern
"""

import pandas as pd

# 1. Datei laden (Wide-Format mit allen Features)
df = pd.read_csv(
    r"\features\v2\features_all_wide.csv"
)

# 2. Kernspalten, die in jedem Fall bleiben (Instanz-ID + globale Geometrie/Skalen)
keep_core = [
    "instance_id",
    "budget",
    "nodes",
    "hull_vertices",
    "hull_area",
    "hull_perimeter",
    "hull_compactness",
    "density",
    "budget_per_node",
]

# 3. Auswahl an Graph-Features für Radius-, kNN-, NNG- und Delaunay-Graphen
keep_builder = [
    # radius-graphs (behaltene)
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

# 4. MST-Features, die als informativ eingestuft wurden
keep_mst = [
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

# Liste von Features, die bewusst verworfen wurden (nur Dokumentation, wird im Code nicht genutzt)
drop_features = [
    "rad_1_0__components",
    "rad_2_0__components",
    "rad_2_0__clustering_avg",
    "rad_2_0__component_avg_size",
    "rad_2_0__component_largest",
    "rad_2_0__degree_avg",
    "rad_2_0__degree_max",
    "rad_2_0__degree_min",
    "knn_3__degree_min",
    "knn_5__degree_min",
    "knn_8__degree_min",
    "knn_12__degree_min",
    "nng_plain__degree_min",
    "nng_mut__degree_min",
    "nng_mut__degree_max",
    "nng_mut__components",
    "nng_mut__component_largest",
    "nng_mut__component_avg_size",
    "nng_mut__clustering_avg",
    "delaunay__components",
    "mst__degree_min",
    "mst__components",
    "mst__clustering_avg",
    "delaunay__component_avg_size",
    "delaunay__component_largest",
    "mst__component_avg_size",
    "mst__component_largest",
    "rad_0_5__component_avg_size",
]

# 5. finale Liste der gewünschten Spalten
keep_cols = keep_core + keep_builder + keep_mst

# 6. Nur Spalten verwenden, die in der eingelesenen Tabelle wirklich existieren
keep_cols = [c for c in keep_cols if c in df.columns]

# 7. reduzierte Feature-Tabelle erzeugen
df_small = df[keep_cols].copy()

# 8. speichern (Trim-Variante im Wide-Format)
df_small.to_csv(
    r"\features\v2\features_all_wide_trim.csv",
    index=False,
)
#df_small.to_parquet(r"\features\v2\features_all_wide_trim.parquet", index=False)

print("Fertig. Behaltene Spalten:", len(keep_cols))
