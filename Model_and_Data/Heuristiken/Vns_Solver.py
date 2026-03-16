"""VNS-Solver für TTDP (Variable Neighborhood Search)

Heuristikaufbau:
- Konstruktion: greedy Insertion (Score / Mehrlänge) bis Budget oder Zeitlimit
- Lokalsuche: zufällige Moves in zwei Nachbarschaften (Relokation / Swap)
              → _vns_local_search mit Level l=1 bzw. l=2
- Shake: zwei Perturbationsoperatoren auf der Route
         * _shake_move_segment       (verschiebt einen Segmentblock)
         * _shake_exchange_segments  (tauscht zwei Segmentblöcke)
- Hauptschleife: klassisches VNS
    * Shake mit Nachbarschaftslvl l
    * lokale Suche in derselben Nachbarschaft
    * akzeptiere nur strikt besseren Score (Budget wird in LS geprüft)
    * bei Verbesserung: l zurücksetzen, sonst l inkrementieren bis l_max
Alle Schleifen respektieren ein hartes Zeitlimit.
"""

from typing import List, Tuple
import random, time
from core import Instance, route_length, route_score
from Ils_Solver import Solution

# globaler RNG, der in den Shake-/LS-Funktionen verwendet wird
rng = None


class VNSParams:
    """Parametercontainer für VNS."""
    def __init__(self, time_limit_s=5.0, seed=0):
        self.time_limit_s = time_limit_s
        self.seed = seed


def construct_initial_vns(inst: Instance, seed: int = 0, t_end: float = float("inf")) -> Solution:
    """
    Baut eine Startlösung für VNS mittels greediger Insertion:

    - Route startet als [start, start]
    - für jeden noch nicht in der Route enthaltenen Knoten:
        * best_insert_for: beste Einfügeposition (minimale Mehrlänge)
        * ratio = score / Mehrlänge
    - wähle jeweils den Kandidaten mit maximalem ratio, solange Budget + Zeitlimit
    """
    global rng
    rng = random.Random(seed)
    n = len(inst.coords)
    route = [inst.start, inst.start]
    in_route = {inst.start}
    cur_len = route_length(inst, route)

    def best_insert_for(node, r):
        """
        Liefert für node die beste Einfügeposition in r:

        Rückgabe:
            (add, pos) mit add = Mehrlänge, pos = Index, nach dem eingefügt wird.
        """
        if time.time() >= t_end:
            return None
        best = None  # (add, pos)
        for pos in range(len(r) - 1):
            if time.time() >= t_end:
                break
            a, b = r[pos], r[pos + 1]
            add = inst.dist[a][node] + inst.dist[node][b] - inst.dist[a][b]
            if add > 1e-12:
                if (best is None) or (add < best[0]):
                    best = (add, pos)
        return best

    # iterativ Kandidaten einfügen, solange sich eine sinnvolle Insertion findet
    while time.time() < t_end:
        cand = []
        for w in range(n):
            if time.time() >= t_end:
                break
            if w in in_route:
                continue
            res = best_insert_for(w, route)
            if res is None:
                continue
            add, pos = res
            # Budgetprüfung auf Rundtour-Länge
            if cur_len + add <= inst.budget + 1e-12:
                ratio = inst.scores[w] / add  # heuristisches Kriterium: Profit / Mehrlänge
                cand.append((ratio, w, pos, add))
        if not cand or time.time() >= t_end:
            break
        cand.sort(key=lambda t: t[0], reverse=True)
        # wähle den besten Kandidaten
        _, w, pos, add = cand[0]
        route.insert(pos + 1, w)
        in_route.add(w)
        cur_len += add

    return Solution(route, cur_len, route_score(inst, route))


def _shake_move_segment(route: List[int]) -> List[int]:
    """
    Shake-Operator 1: verschiebt ein zusammenhängendes Segment an eine andere Stelle.

    - Route wird als geschlossene Tour interpretiert (Start == Ende)
    - es wird ein inneres Segment [i1..i2] ausgewählt
    - dieses Segment wird herausgeschnitten und vor/ hinter einem anderen Index j eingefügt
    """
    global rng
    r = route[:] if route[0] == route[-1] else route[:] + [route[0]]
    m = len(r)
    if m <= 4:  # nur Depot + 1 innerer Knoten
        return r
    i1 = rng.randint(1, m - 3)
    i2 = rng.randint(i1, m - 2)
    # Zielposition außerhalb [i1, i2]
    choices = list(range(1, i1)) + list(range(i2 + 1, m - 1))
    if not choices:
        return r
    j = rng.choice(choices)
    seg = r[i1 : i2 + 1]
    nr = r[:i1] + r[i2 + 1 :]
    j = min(j, len(nr) - 1)
    nr = nr[:j] + seg + nr[j:]
    if nr[0] != nr[-1]:
        nr.append(nr[0])
    return nr


def _shake_exchange_segments(route: List[int]) -> List[int]:
    """
    Shake-Operator 2: tauscht zwei disjunkte Segmente.

    - Route wieder als Rundtour; zwei Intervalle [i1..j1] und [i2..j2]
      werden ausgewählt und vertauscht.
    """
    global rng
    r = route[:] if route[0] == route[-1] else route[:] + [route[0]]
    m = len(r)
    if m <= 6:
        return r
    # zwei disjunkte Intervalle innen wählen
    i1 = rng.randint(1, m - 4)
    j1 = rng.randint(i1, m - 3)
    i2 = rng.randint(j1 + 1, m - 2)
    j2 = rng.randint(i2, m - 2)
    seg1, seg2 = r[i1 : j1 + 1], r[i2 : j2 + 1]
    nr = r[:i1] + seg2 + r[j1 + 1 : i2] + seg1 + r[j2 + 1 :]
    if nr[0] != nr[-1]:
        nr.append(nr[0])
    return nr


def _vns_local_search(inst: Instance, sol: Solution, l: int, t_end: float, trials_per_op: int = None) -> Solution:
    """
    Zufallsbasierte lokale Suche in einer VNS-Nachbarschaft.

    Parameter
    ---------
    l : int
        1 → One-cluster move (Relokation eines Knotens)
        2 → One-cluster exchange (Swap zweier Knoten)
    trials_per_op : int oder None
        Anzahl zufälliger Moves T; Default ~ max(10, n_in^2).

    Akzeptanz:
    - nicht-verschlechternde Moves: Score >= aktueller Score
    - Budgetbedingung muss eingehalten werden
    """
    # trials_per_op ~ n^2 in der Arbeit; hier pragmatisch
    global rng
    r = sol.route[:] if sol.route[0] == sol.route[-1] else sol.route[:] + [sol.route[0]]
    best = Solution(r, route_length(inst, r), route_score(inst, r))
    n_in = max(0, len(r) - 2)
    T = trials_per_op or max(10, n_in * n_in)

    def accept(nr: List[int]) -> bool:
        """Setzt best auf nr, falls Budget erfüllt und Score nicht schlechter."""
        if nr[0] != nr[-1]:
            nr = nr + [nr[0]]
        L = route_length(inst, nr)
        if L > inst.budget + 1e-12:
            return False
        S = route_score(inst, nr)
        # RVNS: nicht-verschlechternd
        if S + 1e-12 >= best.score:
            best.route, best.length, best.score = nr, L, S
            return True
        return False

    # wiederhole den jeweils passenden Operator T-mal oder bis Zeitlimit
    if l == 1:
        # One-cluster move (Relokation eines Knotens)
        for _ in range(T):
            if time.time() >= t_end:
                break
            r = best.route
            m = len(r)
            if m <= 3:
                break
            i1 = rng.randint(1, m - 2)
            i2 = rng.randint(1, m - 2)
            if i1 == i2:
                continue
            nr = r[:]
            node = nr.pop(i1)
            nr.insert(i2, node)
            if accept(nr):
                # Hill-climb: best wird direkt aktualisiert
                pass
    else:
        # One-cluster exchange (Tausch zweier Knoten)
        for _ in range(T):
            if time.time() >= t_end:
                break
            r = best.route
            m = len(r)
            if m <= 3:
                break
            i1 = rng.randint(1, m - 2)
            i2 = rng.randint(1, m - 2)
            if i1 == i2:
                continue
            nr = r[:]
            nr[i1], nr[i2] = nr[i2], nr[i1]
            if accept(nr):
                pass

    return best


def _vns_local_moves(inst: Instance, sol: Solution) -> Solution:
    """
    Alternative deterministische Lokalsuche (nicht im Haupt-VNS verwendet):

    - Relokation: verschiebe einen inneren Knoten an eine andere Position
    - Swap: tausche zwei innere Knoten

    Beide Moves werden exhaustiv ausprobiert, bis kein verbessernder Move
    mehr existiert (klassischer Hill-Climber).
    """
    best = Solution(sol.route[:], sol.length, sol.score)

    def set_if(nr):
        nonlocal best
        # Stelle sicher, dass Route geschlossen ist
        if nr[0] != nr[-1]:
            nr = nr + [nr[0]]
        L = route_length(inst, nr)
        if L > inst.budget + 1e-12:
            return False
        S = route_score(inst, nr)
        # Akzeptiere, wenn Score besser oder gleich mit kürzerer Länge
        if S > best.score + 1e-12 or (abs(S - best.score) <= 1e-12 and L < best.length - 1e-12):
            best = Solution(nr, L, S)
            return True
        return False

    improved = True
    while improved:
        improved = False
        r = best.route
        m = len(r)
        # Relokation
        for i in range(1, m - 1):
            for j in range(1, m - 1):
                if i == j:
                    continue
                nr = r[:]
                node = nr.pop(i)
                nr.insert(j, node)
                if set_if(nr):
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # Swap
        for i in range(1, m - 1):
            for j in range(i + 1, m - 1):
                nr = r[:]
                nr[i], nr[j] = nr[j], nr[i]
                if set_if(nr):
                    improved = True
                    break
            if improved:
                break
    return best


def ttdp_vns(
    points: List[Tuple[float, float, float]],
    start: int,
    budget: float,
    seed: int = 0,
    time_limit_s: float = 5.0,
    l_max: int = 2,
):
    """
    VNS-Hauptfunktion für TTDP.

    Eingabe:
        points : Liste (x, y, score)
        start  : Startknoten (Depotindex)
        budget : Gesamtroutenbudget
        seed   : RNG-Seed
        time_limit_s : hartes Zeitlimit für den gesamten VNS-Lauf
        l_max  : maximale Nachbarschaftsstufe (meist 2)

    Rückgabe:
        (route_nodes, best_score, remaining_budget)
        route_nodes enthält besuchte Knoten ohne abschließenden Depot-Knoten.
    """
    global rng
    inst = Instance.from_points(points, budget, start)
    rng = random.Random(seed)
    t_end = time.time() + time_limit_s

    # Startlösung + erste lokale Suche in beiden Nachbarschaften
    cur = construct_initial_vns(inst, seed, t_end)
    cur = _vns_local_search(inst, cur, l=1, t_end=t_end)
    cur = _vns_local_search(inst, cur, l=2, t_end=t_end)
    # kurze LS-Runde auf beiden Leveln (ursprünglich ohne Zeitlimit; jetzt in t_end integriert)
    best = Solution(cur.route[:], cur.length, cur.score)

    # klassische VNS-Schleife bis Zeitlimit
    while time.time() < t_end:
        l = 1
        improved = False
        while l <= l_max and time.time() < t_end:
            # Shake auf aktueller Lösung (paper-konform)
            if l == 1:
                nr = _shake_move_segment(cur.route)
            else:
                nr = _shake_exchange_segments(cur.route)
            cand = Solution(nr, route_length(inst, nr), route_score(inst, nr))
            # lokale Suche in passender Nachbarschaft
            cand = _vns_local_search(inst, cand, l=l, t_end=t_end)
            # Sicherheitscheck: Budget einhalten
            if route_length(inst, cand.route) > inst.budget + 1e-12:
                l += 1
                continue
            # Akzeptanz: strikt besserer Score (Budget schon in LS geprüft)
            if cand.score > cur.score + 1e-12:
                cur = cand
                if cand.score > best.score + 1e-12 or (
                    abs(cand.score - best.score) <= 1e-12 and cand.length < best.length - 1e-12
                ):
                    best = cand
                l = 1    # nach Verbesserung wieder mit kleinster Nachbarschaft starten
                improved = True
            else:
                l += 1   # keine Verbesserung → nächstgrößere Nachbarschaft
        if not improved:
            # kein Verbesserungssprung in vollständiger VNS-Runde;
            # Abbruch erfolgt nur über Zeitlimit
            pass

    # Route ohne abschließenden Depot-Knoten zurückgeben
    rn = best.route[:-1] if best.route and best.route[0] == best.route[-1] else best.route[:]
    used = route_length(inst, best.route)
    rem = max(0.0, inst.budget - used)
    return rn, best.score, rem
