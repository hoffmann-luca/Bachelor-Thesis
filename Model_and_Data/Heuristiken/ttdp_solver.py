"""ttdp_solvers.py – schlanke Fassade um die eigentlichen TTDP-Heuristiken.

Aufgaben:
- Instanz einlesen (read_points)
- gewählte Heuristik (greedy / ils / vns / grasp) aufrufen
- Lösung optional ausgeben (print_solution_summary) und plotten (plot_ttdp)
"""

from typing import List, Tuple, Optional
from core import read_points, plot_ttdp, print_solution_summary
from Greedy_Solver import greedy_orienteering
from Ils_Solver import ttdp_ils
from Vns_Solver import ttdp_vns
from Grasp_Solver import ttdp_grasp_sr


def quick_run(
    *,
    file: Optional[str] = None,
    points: Optional[List[Tuple[float, float, float]]] = None,
    budget: Optional[float] = None,
    start: int = 0,
    algo: str = "ils",
    seed: int = 0,
    time_limit_s: float = 5.0,
    plot: bool = False,
    verbose: bool = True,
    **plot_kwargs,
) -> Tuple[List[int], float, float]:
    """
    Praktische Fassade für Einzelaufrufe der Heuristiken.

    Entweder:
      - file: Pfad zu einer TTDP-Instanzdatei (Budget wird aus Datei gelesen)
    oder:
      - points + budget: Punkte und Budget direkt übergeben

    Parameter
    ---------
    start : int
        Index des Startknotens (Depot).
    algo : {"greedy","ils","vns","grasp"}
        Zu verwendender Heuristik-Typ.
    seed : int
        Zufalls-Seed (für die stochastischen Verfahren).
    time_limit_s : float
        Zeitbudget in Sekunden, das an die Heuristik weitergereicht wird.
    plot : bool
        Wenn True, wird die gefundene Tour geplottet.
    verbose : bool
        Wenn True, wird eine Textzusammenfassung ausgegeben.
    plot_kwargs :
        Zusätzliche Argumente, die an plot_ttdp weitergereicht werden.

    Returns
    -------
    tour : List[int]
        Besuchte Knoten (ohne zwangsläufig abschließenden Depot-Knoten).
    score : float
        Erreichter Gesamtscore der Tour.
    rem : float
        Verbleibendes Budget.
    """
    # Sicherstellen, dass entweder eine Datei ODER (points+budget) angegeben sind
    assert (file is not None) or (points is not None and budget is not None)

    # Instanz einlesen / Budget setzen
    if points is None:
        points, B = read_points(file)
    else:
        B = float(budget)

    # Algo-Name vereinheitlichen
    a = algo.lower()

    # Entsprechende Heuristik aufrufen
    if a == "greedy":
        tour, score, rem = greedy_orienteering(
            points, start=start, budget=B, time_limit_s=time_limit_s
        )
    elif a == "ils":
        tour, score, rem = ttdp_ils(
            points, start=start, budget=B, seed=seed, time_limit_s=time_limit_s
        )
    elif a == "vns":
        tour, score, rem = ttdp_vns(
            points, start=start, budget=B, seed=seed, time_limit_s=time_limit_s
        )
    elif a == "grasp":
        tour, score, rem = ttdp_grasp_sr(
            points, start=start, budget=B, seed=seed, time_limit_s=time_limit_s
        )
    else:
        # unbekannter Algorithmus-Name
        raise ValueError("algo")

    # Textzusammenfassung der Lösung (Länge, Score, Restbudget, etc.)
    if verbose:
        print_solution_summary(points, B, start, tour)

    # ggf. grafische Darstellung der Tour
    if plot:
        plot_ttdp(points, tour, **plot_kwargs)

    return tour, score, rem


if __name__ == "__main__":
    import os, sys

    # Falls kein CLI-Argument übergeben wurde:
    # - optional TTDP_FILE aus der Umgebung lesen und eine Standard-ILS-Lösung erzeugen
    if len(sys.argv) <= 1:
        env_file = os.getenv("TTDP_FILE")
        if env_file:
            pts, B = read_points(env_file)
            tour, score, rem = ttdp_ils(
                pts, start=0, budget=B, seed=0, time_limit_s=5.0
            )
            print_solution_summary(pts, B, 0, tour)
            # Plot ist hier optional – Fehler beim Plotten werden geschluckt
            try:
                plot_ttdp(pts, tour)
            except Exception:
                pass
            sys.exit(0)
        else:
            print(
                "Keine CLI-Argumente. Nutze quick_run(file=..., ...) oder setze TTDP_FILE."
            )
            sys.exit(0)

    # Regulärer CLI-Modus: Instanzpfad und Optionen werden als Argumente übergeben
    import argparse

    p = argparse.ArgumentParser(description="TTDP Heuristics")
    p.add_argument("file")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--algo", choices=["greedy", "ils", "vns", "grasp"], default="ils")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time", type=float, default=5.0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--annotate", action="store_true", default=True)
    p.add_argument(
        "--annotate_only",
        type=str,
        default="all",
        choices=["all", "tour", "none"],
        help="Welche Punkte beschriftet werden sollen.",
    )
    p.add_argument("--label_fontsize", type=int, default=7)
    p.add_argument(
        "--expand",
        type=float,
        default=1.0,
        help="Optische Streckung im Plot (siehe plotGraph.py).",
    )
    p.add_argument(
        "--jitter",
        type=float,
        default=0.005,
        help="Kleines Jittern der Punkte im Plot zur Entzerrung.",
    )
    args = p.parse_args()

    # Instanz einlesen
    pts, B = read_points(args.file)

    # Gewählte Heuristik ausführen (für CLI noch einmal explizit statt quick_run)
    if args.algo == "greedy":
        tour, score, rem = greedy_orienteering(
            pts, start=args.start, budget=B, time_limit_s=args.time
        )
    elif args.algo == "ils":
        tour, score, rem = ttdp_ils(
            pts, start=args.start, budget=B, seed=args.seed, time_limit_s=args.time
        )
    elif args.algo == "vns":
        tour, score, rem = ttdp_vns(
            pts, start=args.start, budget=B, seed=args.seed, time_limit_s=args.time
        )
    else:
        tour, score, rem = ttdp_grasp_sr(
            pts, start=args.start, budget=B, seed=args.seed, time_limit_s=args.time
        )

    # Textausgabe der gefundenen Lösung
    print_solution_summary(pts, B, args.start, tour)

    # Optionaler Plot der Tour mit den per CLI gesetzten Plot-Parametern
    if args.plot:
        plot_ttdp(
            pts,
            tour,
            expand=args.expand,
            jitter=args.jitter,
            annotate=args.annotate,
            annotate_only=args.annotate_only,
            label_fontsize=args.label_fontsize,
        )
