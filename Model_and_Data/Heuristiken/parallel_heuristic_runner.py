#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-Runner für TTDP-Heuristiken
- Läuft ils/grasp/vns/greedy über viele Instanzen
- Seeds = [0,1,2] (CRN: gleicher Seed je Instanz+Run für alle Algos)
- Zeitlimits konfigurierbar (standard: 1/5/10 s; greedy nur kleinstes Limit)
- Pro Instanz: Algorithmen sequenziell (faire CPU-Bedingungen)
- Parallelisierung über Instanzen via ProcessPoolExecutor
- Warm-up pro (Instanz, Algo) vor Messung (0.1 s)
- CSV-Logging mit Resume (schreibt nur fehlende Kombinationen)

Beispiel:
python run_batch.py --inst_dir ./Trainingsdaten/Artificial/out --csv out/results.csv \
    --algos ils grasp vns greedy --limits 1 5 10 --runs 3 --workers 7 --resume
"""

import os, csv, glob, argparse, time
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from ttdp_solver import quick_run
from core import read_points


# ---------- Utils ----------

def list_instances(inst_dir: str, pattern: str = "*.txt"):
    """
    Listet alle Instanzpfade im Verzeichnis, gefiltert per Glob-Pattern.
    Nur reguläre Dateien werden zurückgegeben.
    """
    paths = sorted(glob.glob(os.path.join(inst_dir, pattern)))
    return [p for p in paths if os.path.isfile(p)]


def instance_id_from_path(path: str) -> str:
    """
    Erzeugt eine Instanz-ID aus dem Dateinamen (ohne Endung).
    """
    return os.path.splitext(os.path.basename(path))[0]


def ensure_csv_header(csv_path: str, header: list):
    """
    Stellt sicher, dass die Ergebnis-CSV existiert und den gewünschten Header hat.

    Falls die Datei noch nicht existiert, wird sie mit einer Header-Zeile angelegt.
    """
    exists = os.path.isfile(csv_path)
    if not exists:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def load_done_set(csv_path: str):
    """
    Liest eine bestehende Ergebnis-CSV ein und baut eine Menge von
    bereits vorhandenen Kombinationen (instance_id, algo, limit_s, run) auf.

    Dient dem Resume-Mechanismus, um bereits berechnete Läufe zu überspringen.
    """
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["instance_id"], row["algo"], float(row["limit_s"]), int(row["run"]))
            done.add(key)
    return done


def limits_by_algo(all_limits, algos):
    """
    Erzeugt ein Mapping algo -> Liste von Limits.

    Aktuell: jeder Algorithmus bekommt dieselben Limits (all_limits).
    """
    by = {}
    for a in algos:
        by[a] = list(all_limits)
    return by


# ---------- Worker ----------

def run_one_instance(file_path: str,
                     start: int,
                     seeds: list,
                     limits_map: dict,
                     warmup_s: float = 0.1):
    """
    Führt alle Algorithmen für EINE Instanz aus und gibt Ergebniszeilen zurück.

    Ablauf:
      - Instanz einlesen
      - Warm-up-Call pro Algorithmus (ohne Logging), um ggf. JIT/Cache-Effekte zu glätten
      - für jeden Seed und jedes Zeitlimit quick_run ausführen
      - Ergebnis (Score, Restbudget, Laufzeit, Tourlänge, …) als Dict-Liste zurückgeben
    """
    inst_id = instance_id_from_path(file_path)
    points, B = read_points(file_path)

    rows = []

    # Warm-up pro Algo (ohne Logging, nur zum "Aufwecken" von Python/NumPy/OS)
    for algo in limits_map.keys():
        try:
            quick_run(points=points, budget=B, start=start,
                      algo=algo, seed=0, time_limit_s=min(0.1, warmup_s),
                      plot=False, verbose=False)
        except Exception:
            # Fehler im Warm-up sind unkritisch und werden ignoriert
            pass

    # eigentliche Runs (Messung)
    for run_idx, seed in enumerate(seeds):
        for algo, lims in limits_map.items():
            for limit in lims:
                t0 = perf_counter()
                tour, score, rem = quick_run(points=points, budget=B, start=start,
                                             algo=algo, seed=seed,
                                             time_limit_s=float(limit),
                                             plot=False, verbose=False)
                dt = perf_counter() - t0
                rows.append({
                    "instance_id": inst_id,
                    "algo": algo,
                    "limit_s": float(limit),
                    "run": run_idx,
                    "seed": int(seed),
                    "score": float(score),
                    "remaining": float(rem),
                    "used_time_s": dt,
                    "n_nodes_in_tour": len(tour),
                    "budget": float(B),
                    "start": int(start),
                })
    return rows


# ---------- Main ----------

def main():
    """
    CLI-Einstiegspunkt:

    - parst Argumente (Instanzordner, Algorithmen, Limits, Seeds, …)
    - erstellt eine Aufgabenliste über Instanzen (unter Berücksichtigung von --resume)
    - verteilt Instanzen auf Worker-Prozesse (je Instanz: alle Algorithmen sequenziell)
    - schreibt Ergebnisse appendend in die CSV
    """
    ap = argparse.ArgumentParser(description="Batch-Runner für TTDP-Heuristiken")
    ap.add_argument("--inst_dir", required=True, help="Ordner mit Instanzdateien (*.txt)")
    ap.add_argument("--csv", required=True, help="Ziel-CSV für Ergebnisse")
    ap.add_argument("--algos", nargs="+", default=["ils", "grasp", "vns", "greedy"],
                    choices=["greedy", "ils", "grasp", "vns"])
    ap.add_argument("--limits", nargs="+", type=float, default=[1.0, 5.0, 10.0],
                    help="Zeitlimits (s). Greedy nutzt standardmäßig nur das kleinste.")
    ap.add_argument("--runs", type=int, default=3, help="Anzahl Seeds/Runs pro Instanz (z. B. 3 → [0,1,2])")
    ap.add_argument("--start", type=int, default=0, help="Startknotenindex")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Parallele Prozesse (über Instanzen)")
    ap.add_argument("--pattern", default="*.txt", help="Dateimuster für Instanzen")
    ap.add_argument("--resume", action="store_true", help="Vorhandene CSV lesen und fertige Läufe überspringen")
    ap.add_argument("--warmup", type=float, default=0.1, help="Warm-up in Sekunden pro (Instanz, Algo), 0.0 = aus")
    args = ap.parse_args()

    # verfügbare Instanzdateien bestimmen
    files = list_instances(args.inst_dir, args.pattern)
    if not files:
        print(f"[!] Keine Instanzen gefunden unter {args.inst_dir} mit Pattern {args.pattern}")
        return

    seeds = list(range(args.runs))  # z. B. [0,1,2]
    lim_map = limits_by_algo(args.limits, args.algos)

    header = ["instance_id", "algo", "limit_s", "run", "seed",
              "score", "remaining", "used_time_s", "n_nodes_in_tour", "budget", "start"]
    ensure_csv_header(args.csv, header)

    # ggf. bereits vorhandene Läufe (Resume) einlesen
    done = load_done_set(args.csv) if args.resume else set()

    # Welche Instanzen müssen gerechnet werden (Resume: nur unvollständige)
    tasks = []
    for f in files:
        inst_id = instance_id_from_path(f)
        all_keys = []
        for r in range(args.runs):
            for a in args.algos:
                for L in lim_map[a]:
                    all_keys.append((inst_id, a, float(L), r))
        # wenn für diese Instanz bereits alle (algo,limit,run)-Kombinationen in der CSV sind → skip
        if args.resume and all(k in done for k in all_keys):
            continue
        tasks.append(f)

    print(f"[i] Instanzen gesamt: {len(files)} | zu rechnen: {len(tasks)}")
    print(f"[i] Algorithmen: {args.algos} | Limits: {args.limits} | Seeds/Runs: {seeds}")
    print(f"[i] Worker: {args.workers} | CSV: {args.csv}")

    t_start = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex, \
         open(args.csv, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        # ein Future pro Instanz (Instanz-level Parallelisierung)
        futs = {ex.submit(run_one_instance, t, args.start, seeds, lim_map, args.warmup): t
                for t in tasks}
        for fu in as_completed(futs):
            inst_path = futs[fu]
            inst_id = instance_id_from_path(inst_path)
            try:
                rows = fu.result()
                # Beim Resume nur Kombinationen schreiben, die noch nicht in 'done' stehen
                for r in rows:
                    key = (r["instance_id"], r["algo"], float(r["limit_s"]), int(r["run"]))
                    if key in done:
                        continue
                    writer.writerow([r[h] for h in header])
                fcsv.flush()
            except Exception as e:
                # Fehler bei einer Instanz protokollieren, Rest weiterlaufen lassen
                print(f"[!] Fehler bei {inst_id}: {e}")
            n_done += 1
            if n_done % 20 == 0:
                elapsed = (time.time() - t_start) / 60.0
                print(f"[i] {n_done}/{len(tasks)} erledigt in {elapsed:.1f} min")

    print("[*] Fertig.")


if __name__ == "__main__":
    main()
