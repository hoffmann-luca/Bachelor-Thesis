#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTDP-Instanzgenerator (.txt-Ausgabe mit Budgetzeile 'tmax <B>' direkt unter 'm 1').

Format der Ausgabedatei '<name>.txt':
n <anzahl_knoten>
m 1
tmax <budget>
x y score          # eine Zeile pro Knoten, erster Knoten = Depot (optional Score 0)

Typische Verwendung:
  - zufällige Punkte (uniform oder clustered) erzeugen
  - diskrete Scores zuweisen (uniform oder Hotspots)
  - Budget entweder direkt vorgeben (--tmax) oder über tau * |V| * d_nn ableiten
  - passende Meta-Information in separater JSON-Datei speichern
"""

from __future__ import print_function
import os, json, math, random, argparse


def euclid(a, b):
    """Euklidische Distanz zwischen zwei 2D-Punkten a=(x,y), b=(x,y)."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def clamp01(x):
    """Beschränkt einen Wert auf das Intervall [0, 1]."""
    return max(0.0, min(1.0, x))


def gen_points_uniform(n, rng):
    """Erzeugt n zufällige Punkte gleichverteilt im Einheitsquadrat [0,1]×[0,1]."""
    return [(rng.random(), rng.random()) for _ in range(n)]


def gen_points_clustered(n, n_clusters, cluster_std, rng):
    """
    Erzeugt n Punkte in Clustern:

    - zunächst werden n_clusters Zufallszentren im Einheitsquadrat gezogen
    - um jedes Zentrum werden Punkte aus einer Normalverteilung mit Std cluster_std erzeugt
    - Koordinaten werden auf [0,1] zurückgeclippt
    """
    centers = [(rng.random(), rng.random()) for _ in range(max(1, n_clusters))]
    pts = []
    for i in range(n):
        cx, cy = centers[i % max(1, n_clusters)]

        def randn():
            # Approximation einer Standardnormalverteilung via Summe von 12 U(0,1)-Variablen
            s = sum(rng.random() for _ in range(12)) - 6.0
            return s

        x = clamp01(cx + cluster_std * randn())
        y = clamp01(cy + cluster_std * randn())
        pts.append((x, y))
    return pts


def gen_scores_uniform(n, a, b, rng):
    """Erzeugt n ganzzahlige Scores ~ U({a, ..., b})."""
    return [rng.randint(a, b) for _ in range(n)]


def gen_scores_hotspots(points, a, b, n_hotspots, bonus, rng):
    """
    Erzeugt Scores mit 'Hotspots':

    - Basisscores ~ U({a, ..., b})
    - wählt n_hotspots Zufallsknoten als Hotspot-Zentren
    - alle Punkte in einem festen Radius um ein Zentrum erhalten einen Bonus
    """
    n = len(points)
    scores = [rng.randint(a, b) for _ in range(n)]
    if n_hotspots <= 0:
        return scores
    centers = rng.sample(range(n), min(n_hotspots, n))
    radius = 0.15
    for i, p in enumerate(points):
        for c in centers:
            if euclid(p, points[c]) <= radius:
                scores[i] += bonus
                break
    return scores


def avg_nearest_neighbor_distance(points):
    """
    Mittlere Distanz zum jeweils nächsten Nachbarn über alle Punkte hinweg.

    Wird zur Skalierung des Budgets (tau * |V| * d_nn) verwendet,
    falls kein explizites --tmax gesetzt ist.
    """
    n = len(points)
    s = 0.0
    for i in range(n):
        mind = float('inf')
        for j in range(n):
            if i == j:
                continue
            d = euclid(points[i], points[j])
            if d < mind:
                mind = d
        s += mind
    return s / float(n)


def save_txt(outdir, basename, points, scores, B):
    """
    Speichert eine Instanz im TTDP-Textformat:

      n <n>
      m 1
      tmax <B>
      x y score      (für jeden Knoten; erster Knoten = Depot)
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{basename}.txt")
    with open(path, "w") as f:
        f.write("n {}\n".format(len(points)))
        f.write("m 1\n")
        f.write("tmax {:.6f}\n".format(B))
        for (x, y), s in zip(points, scores):
            f.write("{:.6f} {:.6f} {}\n".format(x, y, int(s)))
    return path


def save_meta(meta_dir, basename, meta):
    """
    Speichert Meta-Informationen zur Instanz als JSON-Datei
    (gleicher Basename wie .txt, aber mit Endung .json).
    """
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, f"{basename}.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def generate_one(args, seed, outdir, meta_dir, basename):
    """
    Erzeugt genau eine Instanz:

    - generiert Punkte gemäß args.type (uniform/clustered)
    - weist Scores zu (uniform/hotspots)
    - setzt Depot-Score optional auf 0
    - berechnet Budget B (explizit via --tmax oder aus tau * n * d_nn)
    - schreibt .txt-Instanz + Meta-JSON
    """
    rng = random.Random(seed)

    # Punkte generieren
    if args.type == "uniform":
        points = gen_points_uniform(args.n, rng)
    else:
        points = gen_points_clustered(args.n, args.clusters, args.cluster_std, rng)

    # Scores generieren
    if args.scores == "uniform":
        scores = gen_scores_uniform(args.n, args.score_min, args.score_max, rng)
    else:
        scores = gen_scores_hotspots(points, args.score_min, args.score_max,
                                     args.hotspots, args.hotspot_bonus, rng)

    # Depot-Score ggf. auf 0 setzen
    if args.depot_score_zero and len(scores) > 0:
        scores[0] = 0

    # Budget: explizit via --tmax oder über tau * |V| * d_nn
    if args.tmax is not None:
        B = float(args.tmax)
        d_nn = avg_nearest_neighbor_distance(points)  # nur als Meta-Info
    else:
        d_nn = avg_nearest_neighbor_distance(points)
        B = args.tau * float(args.n) * d_nn

    # Instanz + Meta schreiben
    inst_path = save_txt(outdir, basename, points, scores, B)
    save_meta(meta_dir, basename, {
        "name": basename,
        "n": args.n,
        "type": args.type,
        "clusters": args.clusters if args.type == "clustered" else 0,
        "cluster_std": args.cluster_std if args.type == "clustered" else 0.0,
        "scores": args.scores,
        "score_min": args.score_min,
        "score_max": args.score_max,
        "hotspots": args.hotspots if args.scores == "hotspots" else 0,
        "hotspot_bonus": args.hotspot_bonus if args.scores == "hotspots" else 0,
        "tau": args.tau,
        "budget_B": B,
        "d_nn": d_nn,
        "seed": int(seed),
        "depot_score_zero": bool(args.depot_score_zero),
        "format": "TXT: n <n>, m 1, tmax <B>, dann x y score pro Zeile (Depot = erste Zeile)"
    })
    print(f"[ok] {os.path.basename(inst_path)}  (tmax={B:.6f}, seed={seed})")
    return inst_path


def main():
    """
    CLI-Einstiegspunkt:

    - parst Argumente (Ausgabeordner, Instanztyp, Scoring, Budget, Seeds, Batchgröße)
    - erzeugt args.count Instanzen mit fortlaufenden Seeds und Dateinamen
    """
    ap = argparse.ArgumentParser(description="TTDP-Instanzgenerator (.txt-Ausgabe mit tmax)")
    ap.add_argument("--out", type=str, required=True, help="Zielordner für .txt-Instanzen")
    ap.add_argument("--meta", type=str, default=None, help="Ordner für Meta-JSON (Standard: <out>/meta)")
    ap.add_argument("--n", type=int, default=100, help="Anzahl Knoten")
    ap.add_argument("--type", choices=["uniform", "clustered"], default="uniform")
    ap.add_argument("--clusters", type=int, default=4, help="nur für clustered")
    ap.add_argument("--cluster-std", type=float, default=0.08, help="0..1 (nur für clustered)")
    ap.add_argument("--scores", choices=["uniform", "hotspots"], default="uniform")
    ap.add_argument("--score-min", type=int, default=1)
    ap.add_argument("--score-max", type=int, default=10)
    ap.add_argument("--hotspots", type=int, default=0, help="nur bei scores=hotspots")
    ap.add_argument("--hotspot-bonus", type=int, default=3)
    ap.add_argument("--tau", type=float, default=1.2, help="Budgetfaktor; ignoriert, wenn --tmax gesetzt")
    ap.add_argument("--tmax", type=float, default=None, help="Budget direkt setzen; überschreibt tau")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--depot-score-zero",
        action="store_true",
        help="Score des ersten Knotens (Depot) auf 0 setzen"
    )
    # Batch-Parameter
    ap.add_argument("--count", type=int, default=1, help="Anzahl Instanzen (alle in --out, Meta in --meta)")
    ap.add_argument("--prefix", type=str, default="inst", help="Prefix für Dateinamen")
    args = ap.parse_args()

    outdir = args.out
    meta_dir = args.meta if args.meta is not None else os.path.join(outdir, "meta")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # mehrere Instanzen mit unterschiedlichen Seeds und Baser-Namen erzeugen
    for i in range(args.count):
        seed_i = args.seed + i
        basename = f"{args.prefix}_{args.type}_n{args.n}_s{seed_i:04d}"
        generate_one(args, seed_i, outdir, meta_dir, basename)


if __name__ == "__main__":
    main()
