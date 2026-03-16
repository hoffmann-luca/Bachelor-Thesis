# Dieses Skript liest Punktmengen aus .txt-Dateien, berechnet darauf verschiedene Graph-basiere
# Features (Radius-Graph, kNN, Delaunay, NNG, MST, …) und speichert die Ergebnisse
# batchweise als Parquet-Dateien. Optional kann das Ganze parallelisiert werden.


import os, glob, argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

from properties_extractor import (
    PointsDataset, FeaturePipeline,
    build_radius_graph, build_knn_graph,
    build_delaunay_graph, build_mst_via_delaunay, build_nng_via_delaunay,
    feature_basic_counts, feature_connected_components, feature_clustering,
    feature_hull_stats, feature_mst_stats, feature_mst_norms_both, _delaunay_tris, _delaunay_edge_set_from_tris
)

import pyarrow as pa
import pyarrow.parquet as pq
from itertools import count

def append_parquet_batch(batch_frames, out_dir: str, part_counter=[count(start=0)]):
    """
    Fasst einen Batch von Feature-DataFrames zusammen und schreibt ihn
    als separate Parquet-Datei in ein Ausgabeverzeichnis.
    Der Dateiname enthält eine fortlaufende Partitionsnummer.
    """
    if not batch_frames:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat(batch_frames, ignore_index=True)
    tid = next(part_counter[0])
    path = out / f"features-part-{tid:05d}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


# Parameter der geometrischen/graph-basierten Konstruktionen
QHULL = "QJ"               # Optionen für qhull bei Delaunay-Berechnungen
R_LIST = (0.5, 1.0, 2.0)   # getestete Radien für Radius-Graphen
K_LIST = (3, 5, 8, 12)     # getestete k-Werte für kNN-Graphen


def one_file(path: str) -> pd.DataFrame:
    """
    Berechnet alle konfigurierten Graph-Features für eine einzelne Punktdatei.
    Liefert einen DataFrame mit mehreren Zeilen pro Datei
    (je eine Zeile pro Graph-Konstruktion / Builder).
    """
    ds = PointsDataset.from_file(path)
    # globale Datensatz-Features aus der konvexen Hülle
    hull = feature_hull_stats(ds, g=None, qhull_options=QHULL)

    def features_for(builder, **kwargs):
        """
        Führt eine FeaturePipeline für einen gegebenen Graph-Builder aus
        (z.B. Radius-, kNN-, Delaunay-, NNG-Graph) und versieht das Ergebnis
        mit Datensatz- und Builder-Metadaten.
        """
        feats = [feature_basic_counts, feature_connected_components, feature_clustering]
        pipe = FeaturePipeline(builder, *feats, **kwargs)
        out = pipe.run(ds)
        out.update(hull)              # globale Datensatz-Statistiken (z.B. Hüllvolumen)
        out.update(kwargs)            # Hyperparameter wie radius/k/mutual/qhull_options
        out["builder"] = builder.__name__
        return out

    results = []

    # Radius-Graphen für verschiedene Radien
    for r in R_LIST:
        results.append(features_for(build_radius_graph, radius=r))

    # kNN-Graphen für verschiedene k
    for k in K_LIST:
        results.append(features_for(build_knn_graph, k=k))

    # Delaunay einmal aus den Koordinaten berechnen und Kanten daraus ableiten,
    # um sie für verschiedene Graph-Varianten wiederverwenden zu können.
    tris = _delaunay_tris(ds.coords, qhull_options="QJ")
    edges = _delaunay_edge_set_from_tris(tris)

    # NNG (Nearest-Neighbor-Graph), einmal allgemeiner und einmal nur wechselseitige Nachbarschaft
    results.append(features_for(build_nng_via_delaunay, qhull_options=QHULL, mutual=False, precomputed_edges=edges))
    results.append(features_for(build_nng_via_delaunay, qhull_options=QHULL, mutual=True, precomputed_edges=edges))

    # Delaunay-Graph auf Basis der vorab berechneten Kanten
    results.append(features_for(build_delaunay_graph, qhull_options=QHULL, precomputed_edges=edges))

    # MST (Minimum Spanning Tree) inkl. spezieller MST-Statistiken und Normierungen
    mst_feats = [feature_basic_counts, feature_connected_components, feature_clustering,
                 feature_mst_stats, feature_mst_norms_both]
    pipe_mst = FeaturePipeline(build_mst_via_delaunay, *mst_feats, qhull_options=QHULL, precomputed_edges=edges)
    out_mst = pipe_mst.run(ds)
    out_mst.update(hull)
    out_mst.update({"builder": "build_mst_via_delaunay", "qhull_options": QHULL})
    results.append(out_mst)

    df = pd.DataFrame(results)
    # Identität und Budget-Informationen des Datensatzes ergänzen
    df["dataset_id"] = os.path.splitext(os.path.basename(path))[0]
    df["budget"] = ds.budget
    df["budget_per_node"] = ds.budget / ds.n
    df["source_path"] = path

    # Wichtige Spalten nach vorne ziehen, Rest hinten anhängen
    front = [c for c in ["dataset_id", "builder", "radius", "k", "budget", "mutual", "qhull_options"] if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


def append_csv_batch(batch_frames, out_csv: str):
    """
    Alternative Ausgabefunktion: verteilt berechnete Feature-DataFrames batchweise
    an eine CSV-Datei (Append-Modus). Wird im aktuellen Setup nicht genutzt.
    """
    if not batch_frames:
        return
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat(batch_frames, ignore_index=True)
    header = not out.exists()
    df.to_csv(out, mode="a", header=header, index=False, encoding="utf-8")


def main():
    """
    Steuert den Batch-Prozess:
    - sammelt alle .txt-Dateien im Eingabeordner ein,
    - berechnet deren Graph-Features (seriell oder parallel),
    - schreibt die Ergebnisse batchweise als Parquet-Dateien.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Ordner mit .txt-Dateien")
    ap.add_argument("--out_parquet_dir", required=True, help="Ziel-CSV (wird angehängt)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2),
                    help="Anzahl Prozesse (1 = ohne Parallelisierung)")
    ap.add_argument("--batch_size", type=int, default=100, help="Anzahl Dateien pro Schreibbatch")
    ap.add_argument("--log_every", type=int, default=200, help="Statusmeldung alle N Dateien")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.txt")))
    if not files:
        print("Keine .txt-Dateien gefunden.")
        return

    t0 = time.time()
    total = len(files)
    processed = 0

    print(f"[*] Starte auf {len(files)} Dateien | workers={args.workers} | batch={args.batch_size}")

    batch = []
    if args.workers == 1:
        # rein serielle Verarbeitung der Dateien
        for i, p in enumerate(files, 1):
            batch.append(one_file(p))
            processed += 1

            # Fortschritts-Logging inkl. Durchsatz und geschätzter Restzeit
            if processed % args.log_every == 0 or processed == total:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (total - processed) / rate if rate > 0 else float('inf')
                print(f"[info] {processed}/{total} | {rate:.2f} files/s | elapsed={elapsed:.1f}s | ETA={eta:.1f}s")

            if len(batch) >= args.batch_size:
                append_parquet_batch(batch, args.out_parquet_dir)
                print(f"[write] batch={len(batch)} → parquet | done={processed}/{total}")
                batch.clear()

        if batch:
            # letzten unvollständigen Batch noch schreiben
            append_parquet_batch(batch, args.out_parquet_dir)
            print(f"[write] final batch={len(batch)} → parquet | done={processed}/{total}")
            batch.clear()
    else:
        # parallele Verarbeitung über mehrere Prozesse
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(one_file, p): p for p in files}
            for fut in as_completed(futures):
                try:
                    batch.append(fut.result())
                except Exception as e:
                    # fehlerhafte Datei protokollieren, aber den Gesamtprozess fortsetzen
                    print(f"[warn] Datei fehlerhaft: {futures[fut]} ({e})")
                processed += 1

                if processed % args.log_every == 0 or processed == total:
                    elapsed = time.time() - t0
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    eta = (total - processed) / rate if rate > 0 else float('inf')
                    print(f"[info] {processed}/{total} | {rate:.2f} files/s | elapsed={elapsed:.1f}s | ETA={eta:.1f}s")

                if len(batch) >= args.batch_size:
                    append_parquet_batch(batch, args.out_parquet_dir)
                    print(f"[write] batch={len(batch)} → parquet | done={processed}/{total}")
                    batch.clear()

            if batch:
                append_parquet_batch(batch, args.out_parquet_dir)
                print(f"[write] final batch={len(batch)} → parquet | done={processed}/{total}")
                batch.clear()

    print("[*] Fertig.")


if __name__ == "__main__":
    main()