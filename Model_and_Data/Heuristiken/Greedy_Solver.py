"""Greedy-Baseline für TTDP

Baut eine geschlossene Route [start, ..., start], indem jeweils der beste noch
nicht besuchte Knoten eingefügt wird.

- für jeden Kandidatenknoten wird über alle Einfügepositionen die Mehrlänge ΔC
  der Route berechnet
- der Knoten mit maximalem Verhältnis (Score / ΔC) wird gewählt
- Einfügen erfolgt nur, wenn die entstehende Rundtour das Budget nicht verletzt
- es gibt eine harte Laufzeitbeschränkung time_limit_s für die gesamte Heuristik
"""

from typing import List, Tuple
import math, time


def greedy_orienteering(points: List[Tuple[float, float, float]],
                        start: int = 0,
                        budget: float = 20.0,
                        time_limit_s: float = 5.0):
    """
    Greedy-Heuristik für eine TTDP-Instanz.

    Parameters
    ----------
    points : Liste von (x, y, score)
    start  : Index des Depotknotens
    budget : maximale zulässige Routendistanz
    time_limit_s : maximales Zeitbudget für den Greedy-Aufbau

    Returns
    -------
    route_nodes : Liste von Knoten-IDs ohne abschließendes Depot
    profit      : erreichte Gesamtsumme der Scores (einmal pro besuchtem Knoten)
    remaining   : verbleibendes Distanzbudget (>= 0)
    """
    n = len(points)
    coords = [(x, y) for (x, y, _) in points]
    scores = [s for (_, _, s) in points]

    def dist(i: int, j: int) -> float:
        """Euklidische Distanz zwischen Knoten i und j."""
        (xi, yi), (xj, yj) = coords[i], coords[j]
        return math.hypot(xi - xj, yi - yj)

    # Route immer als Rundtour [start, ..., start] halten
    route = [start, start]
    in_route = {start}

    # absolute Zeitgrenze für die gesamte Greedy-Konstruktion
    t_end = time.time() + time_limit_s

    while True:
        if time.time() >= t_end:
            # Zeitlimit erreicht → Konstruktion abbrechen
            break

        best = None  # (ratio, node, min_add, best_pos)

        # aktuelle Routenlänge L = Summe der Kanten in 'route'
        L = 0.0
        for i in range(len(route) - 1):
            L += dist(route[i], route[i + 1])

        # über alle noch nicht besuchten Knoten iterieren
        for node in range(n):
            if time.time() >= t_end:
                break
            if node in in_route:
                continue

            # beste Einfügeposition (über alle Kanten der aktuellen Route)
            min_add = float('inf')
            best_pos = None
            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                add = dist(a, node) + dist(node, b) - dist(a, b)
                if add < min_add:
                    min_add = add
                    best_pos = i

            # nur zulässige Einfügungen betrachten (Budgetbedingung)
            if L + min_add <= budget + 1e-12:
                # Score / Mehrkosten als Auswahlkriterium
                denom = min_add if min_add > 1e-12 else 1e-12
                ratio = scores[node] / denom
                if (best is None) or (ratio > best[0]):
                    best = (ratio, node, min_add, best_pos)

        # kein zulässiger Kandidat mehr → Greedy-Konstruktion fertig
        if best is None:
            break

        # besten Kandidaten tatsächlich in die Route einfügen
        _, node, min_add, best_pos = best
        route.insert(best_pos + 1, node)
        in_route.add(node)

    # Endauswertung der konstruierten Rundtour
    used = 0.0
    for i in range(len(route) - 1):
        used += dist(route[i], route[i + 1])
    # Score jedes besuchten Knotens zählt genau einmal (Depot ausgenommen)
    profit = sum(scores[i] for i in set(route) if i != start)
    remaining = max(0.0, budget - used)

    # Route ohne abschließendes Depot zurückgeben
    return route[:-1], profit, remaining
