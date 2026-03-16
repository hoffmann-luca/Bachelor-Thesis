"""TTDP Core Utilities

Gemeinsame Hilfsfunktionen/Strukturen für alle Heuristiken:
- Einlesen der Instanz und Aufbau der Distanzmatrix
- Normalisierung von Routen
- Berechnung von Routenlänge und -profit
- Plot-Helfer (nutzt ggf. plotGraph)
- kompakte Lösungs-Summary
"""
from typing import List, Tuple
import math


# ------------------ I/O ------------------

def read_points(file_path: str) -> Tuple[List[Tuple[float, float, float]], float]:
    """
    Liest eine Punktdatei im TTDP-Format.

    Erwartetes Format:
      1.–2. Zeile: Header (ignoriert)
      3. Zeile:    <label> <budget>
      ab 4. Zeile: x y score

    Returns
    -------
    pts : Liste von Tupeln (x, y, score)
    B   : Budget (float)
    """
    pts: List[Tuple[float, float, float]] = []
    B = 0.0
    k = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            # erste beiden Zeilen ignorieren
            if k <= 1:
                k += 1
                continue
            # dritte Zeile: Budget
            if k == 2:
                if line:
                    p = line.split()
                    if len(p) >= 2:
                        try:
                            B = float(p[-1])
                        except Exception:
                            B = 0.0
                k += 1
                continue
            # ab Zeile 4: Punkte
            if not line:
                k += 1
                continue
            p = line.split()
            if len(p) >= 3:
                pts.append((float(p[0]), float(p[1]), float(p[2])))
            k += 1
    return pts, B


# -------------- Instanz/Distanz --------------

class Instance:
    """
    TTDP-Instanz mit vorab berechneter euklidischer Distanzmatrix.

    Attributes
    ----------
    coords : Liste von (x, y)
    scores : Score je Knoten
    budget : maximal zulässige Routenlänge
    start  : Start-/Depotindex
    dist   : Distanzmatrix dist[i][j]
    """
    def __init__(self, coords, scores, budget, start=0, dist=None):
        self.coords = coords
        self.scores = scores
        self.budget = budget
        self.start = start
        self.dist = dist

    @staticmethod
    def from_points(points: List[Tuple[float, float, float]], budget: float, start: int = 0) -> "Instance":
        """
        Baut aus (x, y, score)-Punkten eine Instanz und berechnet
        die vollständige euklidische Distanzmatrix.
        """
        coords = [(x, y) for (x, y, _) in points]
        scores = [s for (_, _, s) in points]
        n = len(points)
        D = [[0.0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                d = math.hypot(xi - xj, yi - yj)
                D[i][j] = D[j][i] = d
        return Instance(coords, scores, budget, start, D)


# -------------- Route-Utilities --------------

def _normalize_route(route: List[int], start: int) -> List[int]:
    """
    Normalisiert eine Route, sodass:

    - aufeinanderfolgende Duplikate entfernt sind,
    - die Route mit dem Startknoten beginnt,
    - der Startknoten im Inneren nicht vorkommt,
    - die Route am Ende wieder zum Start zurückführt.

    Ergebnisform: [start, ..., start]
    """
    if not route:
        return [start, start]
    # benachbarte Duplikate entfernen
    r = [route[0]]
    for v in route[1:]:
        if v != r[-1]:
            r.append(v)
    # sicherstellen, dass die Route mit start beginnt
    if r[0] != start:
        r = [start] + r
    # inneren Start-Knoten entfernen
    inner = [v for v in r[1:] if v != start]
    r = [start] + inner
    # Route schließen
    if r[-1] != start:
        r.append(start)
    return r


def route_length(inst: Instance, route: List[int]) -> float:
    """
    Berechnet die Gesamtlänge einer Route (nach Normalisierung)
    anhand der Distanzmatrix der Instanz.
    """
    if not route:
        return 0.0
    r = _normalize_route(route, inst.start)
    tot = 0.0
    for i in range(len(r) - 1):
        tot += inst.dist[r[i]][r[i + 1]]
    return tot


def route_score(inst: Instance, route: List[int]) -> float:
    """
    Berechnet den Gesamtprofit einer Route.

    - Depot/Startknoten wird nicht gezählt.
    - Jeder Knoten trägt seinen Score höchstens einmal bei,
      auch wenn er mehrfach in der Route vorkommt.
    """
    if not route:
        return 0.0
    r = _normalize_route(route, inst.start)
    seen = set(r)
    seen.discard(inst.start)
    return float(sum(inst.scores[i] for i in seen))


# ------------------ Plot ------------------
try:
    # Versuch: plotGraph als Modul importieren (gleiche Struktur wie in Greedy-Umgebung)
    import plotGraph as _plotGraph
except Exception:
    try:
        import plotGraph as _plotGraph
    except Exception:
        # wenn kein plotGraph vorhanden ist, auf Matplotlib-Fallback gehen
        _plotGraph = None


def _build_plot_sets(points: List[Tuple[float, float, float]], tour_nodes: List[int]):
    """
    Zerlegt die Punkteliste in
    - tourPts: Koordinaten der Punkte, die in der Tour vorkommen
    - otherPts: alle übrigen Punkte.

    Das erleichtert die farbliche Trennung im Plot.
    """
    nodes = {i: (x, y) for i, (x, y, _) in enumerate(points)}
    tourPts = []
    for j in tour_nodes:
        if j in nodes:
            tourPts.append(nodes.pop(j))
    otherPts = list(nodes.values())
    return tourPts, otherPts


def plot_ttdp(points: List[Tuple[float, float, float]], tour_nodes: List[int], *,
              expand: float = 1.0, jitter: float = 0.005,
              annotate: bool = True, annotate_only: str = "all", label_fontsize: int = 7) -> None:
    """
    Visualisiert eine TTDP-Tour.

    Falls ein plotGraph-Modul verfügbar ist, wird dessen plotTSP-Funktion benutzt.
    Andernfalls erfolgt ein einfacher Matplotlib-Fallback.

    Parameters
    ----------
    points : Liste von (x, y, score)
    tour_nodes : Folge von Knotenindizes in Besuchsreihenfolge (ohne Enddepot)
    expand, jitter, annotate, annotate_only, label_fontsize :
        werden an plotGraph bzw. an den Matplotlib-Fallback weitergereicht.
    """
    tourPts, otherPts = _build_plot_sets(points, tour_nodes)
    # bevorzugt plotGraph nutzen, wenn vorhanden
    if _plotGraph is not None and hasattr(_plotGraph, "plotTSP"):
        _plotGraph.plotTSP(
            tourPts,
            otherPts,
            points,
            expand=expand,
            jitter=jitter,
            annotate=annotate,
            annotate_only=annotate_only,
            label_fontsize=label_fontsize,
        )
        return
    # Fallback: einfache Darstellung mit Matplotlib
    try:
        import random as _rnd
        import matplotlib.pyplot as plt  # type: ignore

        def _jit(x, y):
            return (x + _rnd.uniform(-jitter, jitter), y + _rnd.uniform(-jitter, jitter)) if jitter else (x, y)

        if otherPts:
            xs, ys = zip(*[_jit(x, y) for (x, y) in otherPts])
            plt.scatter(xs, ys, marker="o", alpha=0.7)

        if tourPts:
            # Tour als Linie (geschlossen) darstellen
            loop = tourPts + [tourPts[0]] if len(tourPts) >= 2 else tourPts
            xs, ys = zip(*[_jit(x, y) for (x, y) in loop])
            plt.plot(xs, ys, linewidth=1.5)
            plt.scatter(xs[:-1], ys[:-1])

        if annotate and points:
            if annotate_only in ("all", "tour"):
                on_tour = set(tour_nodes)
                for idx, (x, y, _) in enumerate(points):
                    if annotate_only == "tour" and idx not in on_tour:
                        continue
                    plt.annotate(str(idx), _jit(x, y), fontsize=label_fontsize)

        plt.title("TTDP-Lösung")
        plt.axis("equal")
        plt.show()
    except Exception as e:
        # bewusst nur einfache Konsolenmeldung, kein erneuter Fallback
        print("Plot-Fallback fehlgeschlagen:", e)


# ------------------ Summary ------------------

def solution_summary(points, budget, start, tour_nodes):
    """
    Baut eine kompakte Lösungsbeschreibung (Dict) für eine Tour.

    tour_nodes enthält die besuchten Knoten (ohne explizites Enddepot),
    start ist der Index des Depots.

    Returns
    -------
    dict mit Schlüsseln:
      - max_budget  : vorgegebenes Budget
      - distance    : verwendete Routenlänge
      - visited     : Liste der besuchten Knoten (ohne Enddepot)
      - profit      : erzielter Gesamtprofit
      - remaining   : Restbudget (max(0, Budget − Distanz))
    """
    inst = Instance.from_points(points, budget, start)
    route = tour_nodes[:]
    # sicherstellen, dass Route mit start beginnt
    if route and (route[0] != start):
        route = [start] + route
    # und dass sie zum Start zurückkehrt
    if (not route) or route[-1] != start:
        route = route + [start]
    used = route_length(inst, route)
    score = route_score(inst, route)
    return {
        "max_budget": float(budget),
        "distance": float(used),
        "visited": tour_nodes[:],
        "profit": float(score),
        "remaining": float(max(0.0, budget - used)),
    }


def print_solution_summary(points, budget, start, tour_nodes):
    """
    Gibt die von solution_summary berechneten Kennzahlen formatiert auf stdout aus.
    """
    s = solution_summary(points, budget, start, tour_nodes)
    print("Max Budget:", s["max_budget"])
    print("Zurückgelegte Distanz:", s["distance"])
    print("Besuchte Punkte (IDs):", s["visited"])
    print("Gesamtprofit:", s["profit"])
    print("Restbudget:", s["remaining"])
