# prepare_results.py
import pandas as pd
from pathlib import Path

# Pfad zu den Eingabedateien und Ausgabedateien
# FEATURES_PATH: enthält die Features pro Instanz
# RESULTS_PATH: enthält die Ergebnisse der Algorithmenläufe
# OUT_AGG_PATH: Zwischendatei mit aggregierten Ergebnissen pro (Instanz, Limit, Algorithmus)
# OUT_MERGED: finale Trainingsdatendatei mit Features und Zielvariablen
FEATURES_PATH = Path(r"\features\v2\features_all_wide_trim.csv")
RESULTS_PATH  = Path(r"\features\v2\result_all.csv")
OUT_AGG_PATH  = Path(r"\features\v2\results_agg.csv")
OUT_MERGED    = Path(r"\features\v2\train_data.csv")

def main():
    # Einlesen der Feature- und Ergebnisdaten aus CSV-Dateien
    feats = pd.read_csv(FEATURES_PATH)
    res = pd.read_csv(RESULTS_PATH)

    # 1) pro (inst, limit, algo) mitteln
    #    - Gruppierung nach Instanz, Zeitlimit und Algorithmus
    #    - Berechnung von mittlerem Score, mittlerem Restbudget, mittlerer Laufzeit
    #      sowie der Anzahl der Runs (über Seeds)
    agg = (
        res.groupby(["instance_id", "limit_s", "algo"], as_index=False)
          .agg(
              mean_score=("score", "mean"),              # durchschnittlicher Score
              mean_remaining=("remaining", "mean"),      # durchschnittlich verbleibende Zeit/Resourcen
              mean_used_time_s=("used_time_s", "mean"),  # durchschnittlich genutzte Zeit
              n_runs=("seed", "count"),                  # Anzahl der Runs (Seeds)
          )
    )
    # Speichern der aggregierten Ergebnisse
    agg.to_csv(OUT_AGG_PATH, index=False)

    # 2) bestes Verfahren pro (inst, limit) bestimmen
    #    - Sortierung innerhalb jeder (inst, limit)-Gruppe nach:
    #        * mean_score (absteigend: höher = besser)
    #        * mean_remaining (absteigend: mehr Rest = besser, falls Score gleich)
    #    - Danach wird je Gruppe die erste Zeile genommen = der beste Algorithmus
    best = (
        agg.sort_values(
            ["instance_id", "limit_s", "mean_score", "mean_remaining"],
            ascending=[True, True, False, False]
        )
        .groupby(["instance_id", "limit_s"], as_index=False)
        .first()                           # wählt je Gruppe den "besten" Eintrag
        .rename(columns={"algo": "best_algo"})  # Umbenennung der Algorithmus-Spalte
    )

    # 3) Features dazu mergen → eine Zeile pro (inst, limit)
    #    - Verknüpft die beste Algorithmuswahl je (inst, limit) mit den zugehörigen Features
    #    - Join erfolgt über die Instanz-ID
    merged = best.merge(feats, on="instance_id", how="left")

    # Speichern der finalen Trainingsdaten (Features + Zielinformationen)
    merged.to_csv(OUT_MERGED, index=False)
    print(f"[ok] geschrieben: {OUT_AGG_PATH} und {OUT_MERGED}")

# Einstiegspunkt: führt main() aus, wenn das Skript direkt gestartet wird
if __name__ == "__main__":
    main()
