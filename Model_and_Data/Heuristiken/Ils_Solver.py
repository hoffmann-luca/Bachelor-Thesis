from typing import List, Tuple, Optional
import random, time
from core import Instance, route_length, route_score

rng = None


class ILSParams:
    """Hyperparameter für ILS (Zeitlimit, Zufall, Intensität der Suche)."""
    def __init__(self, time_limit_s=5.0, seed=0, f_keep=30, threshold_intensify=20, max_no_improve_ls=30):
        self.time_limit_s = time_limit_s
        self.seed = seed
        self.f_keep = f_keep                      # Größe der Kandidatenliste (Top-F-Lösungen) bei Insertion
        self.threshold_intensify = threshold_intensify  # nach wie vielen erfolglosen Iterationen zur Bestlösung zurückspringen
        self.max_no_improve_ls = max_no_improve_ls      # Abbruch der Lokalsuche nach so vielen erfolglosen Moves


class Solution:
    """Einfacher Lösungstyp: Route (inkl. Depot), Routenlänge, Score."""
    def __init__(self, route, length, score):
        self.route = route
        self.length = length
        self.score = score


# ---------- Hilfsfunktionen ----------

def delta_insert_cost(inst: Instance, route: List[int], pos: int, node: int) -> float:
    """
    Mehrlänge beim Einfügen von node hinter route[pos].

    Berechnet die Kostendifferenz, wenn die Kante (route[pos], route[pos+1])
    durch zwei Kanten (route[pos], node) und (node, route[pos+1]) ersetzt wird.
    """
    a = route[pos]
    b = route[(pos + 1) % len(route)]
    return inst.dist[a][node] + inst.dist[node][b] - inst.dist[a][b]


def feasible_after_insert(inst: Instance, route: List[int], pos: int, node: int, current_len: Optional[float] = None):
    """
    Prüft, ob das Einfügen von node an Position pos das Budget einhält.

    Gibt (feasible, delta) zurück:
        feasible : True/False bzgl. Budget
        delta    : Mehrlänge durch die Insertion
    """
    if current_len is None:
        current_len = route_length(inst, route)
    add = delta_insert_cost(inst, route, pos, node)
    return (current_len + add <= inst.budget + 1e-12), add


# ---------- Konstruktion ----------

def construct_initial(inst: Instance, params: ILSParams, t_end: float) -> Solution:
    """
    Konstruiert eine Startlösung durch stochastische Insertion:

    - Route beginnt als [start, start]
    - Kandidaten sind alle noch nicht besuchten Knoten, die budget-feasible insertierbar sind
    - Bewertung mit ratio = score^2 / Δlänge
    - Auswahl per Roulette-Wheel aus den besten F-Kandidaten (f_keep)
    - bricht ab, wenn keine Kandidaten mehr oder Zeitlimit erreicht ist
    """
    global rng
    rng = random.Random(params.seed)
    n = len(inst.coords)
    route = [inst.start, inst.start]
    in_route = {inst.start}
    cur_len = 0.0

    def build_F():
        """
        Baut die Insertion-Kandidatenliste F:
        Eintrag: (ratio, node, pos, add)
        """
        F = []
        for node in range(n):
            if time.time() >= t_end:
                break
            if node in in_route:
                continue
            for pos in range(len(route) - 1):
                if time.time() >= t_end:
                    break
                feas, add = feasible_after_insert(inst, route, pos, node, cur_len)
                if not feas or add <= 0.0:
                    continue
                ratio = (inst.scores[node] ** 2) / add
                if ratio > 0:
                    F.append((ratio, node, pos, add))
        F.sort(key=lambda t: t[0], reverse=True)
        if params.f_keep > 0 and len(F) > params.f_keep:
            F = F[:params.f_keep]
        return F

    while time.time() < t_end:
        F = build_F()
        if not F or time.time() >= t_end:
            break

        # Roulette-Wheel-Auswahl auf Basis der ratio-Werte
        s = sum(r for (r, _, _, _) in F)
        u = rng.random() * s
        acc = 0.0
        chosen = F[-1]
        for it in F:
            acc += it[0]
            if u <= acc:
                chosen = it
                break
        _, node, pos, add = chosen
        route.insert(pos + 1, node)
        in_route.add(node)
        cur_len += add

    return Solution(route, cur_len, route_score(inst, route))


# ---------- Lokalsuche ----------

def local_search(inst: Instance, sol: Solution, p: ILSParams, t_end: float) -> Solution:
    """
    Lokalsuche auf einer gegebenen Lösung:

    Neighborhoods:
      - Swap: Vertauschen zweier innerer Knoten
      - 2-opt: Teilpfad umdrehen
      - Insert: bisher unbesuchte Knoten budget-feasible einfügen
      - Replace: Knoten auf Route durch besserscored Knoten ersetzen

    Akzeptanzregel: neuer Score besser oder Score gleich & Route kürzer.
    Abbruch, wenn max_no_improve_ls Moves ohne Verbesserung oder Zeitlimit erreicht ist.
    """
    n = len(inst.coords)
    best = Solution(sol.route[:], sol.length, sol.score)
    no_imp = 0

    def apply(new_r):
        """Akzeptiert new_r nur, wenn Budget erfüllt und Verbesserung gegenüber best."""
        nonlocal best, no_imp
        if new_r[0] != new_r[-1]:
            new_r = new_r + [new_r[0]]
        L = route_length(inst, new_r)
        if L > inst.budget + 1e-12:
            return False
        S = route_score(inst, new_r)
        if S > best.score + 1e-12 or (abs(S - best.score) <= 1e-12 and L < best.length - 1e-12):
            best = Solution(new_r, L, S)
            no_imp = 0
            return True
        return False

    def unscheduled(r):
        """
        Liefert die Knoten, die aktuell nicht in der Route liegen.

        Depot wird immer als "in Route" betrachtet.
        """
        s = set(r)
        s.add(inst.start)
        return [i for i in range(n) if i not in s]

    while no_imp < p.max_no_improve_ls and time.time() < t_end:
        if time.time() >= t_end:
            break
        improved = False
        r = best.route
        m = len(r)

        # Swap-Nachbarschaft
        for i in range(1, m - 1):
            if time.time() >= t_end:
                break
            for j in range(i + 1, m - 1):
                if time.time() >= t_end:
                    break
                nr = r[:]
                nr[i], nr[j] = nr[j], nr[i]
                if apply(nr):
                    r = best.route
                    m = len(r)
                    improved = True
                    break
            if improved:
                break
        if improved or time.time() >= t_end:
            continue

        # 2-opt-Nachbarschaft
        for i in range(1, m - 2):
            if time.time() >= t_end:
                break
            for j in range(i + 1, m - 1):
                if time.time() >= t_end:
                    break
                nr = r[:i] + list(reversed(r[i:j + 1])) + r[j + 1:]
                if apply(nr):
                    r = best.route
                    m = len(r)
                    improved = True
                    break
            if improved:
                break
        if improved or time.time() >= t_end:
            continue

        # Insert: unbesuchte Knoten einfügen
        uns = unscheduled(r)
        if uns:
            F = []
            cur_len = best.length
            for node in uns:
                if time.time() >= t_end:
                    break
                for pos in range(len(r) - 1):
                    if time.time() >= t_end:
                        break
                    feas, add = feasible_after_insert(inst, r, pos, node, cur_len)
                    if feas and add > 0:
                        ratio = (inst.scores[node] ** 2) / add
                        if ratio > 0:
                            F.append((ratio, node, pos, add))
            if F:
                F.sort(key=lambda t: t[0], reverse=True)
                if p.f_keep > 0 and len(F) > p.f_keep:
                    F = F[:p.f_keep]
                s = sum(x[0] for x in F)
                if time.time() >= t_end or s <= 0:
                    break
                u = rng.random() * s
                acc = 0.0
                chosen = F[-1]
                for it in F:
                    acc += it[0]
                    if u <= acc:
                        chosen = it
                        break
                _, node, pos, add = chosen
                nr = r[:]
                nr.insert(pos + 1, node)
                if apply(nr):
                    improved = True
                    continue
        if improved:
            continue

        # Replace: Route-Knoten durch unbesuchte besser scorende Knoten ersetzen
        uns = unscheduled(r)
        if uns:
            uns_sorted = sorted(uns, key=lambda i: inst.scores[i], reverse=True)
            replaced = False
            if time.time() >= t_end:
                break
            for cand in uns_sorted:
                if time.time() >= t_end:
                    break
                for pos in range(1, len(r) - 1):
                    if time.time() >= t_end:
                        break
                    old = r[pos]
                    if inst.scores[cand] <= inst.scores[old] + 1e-12:
                        continue
                    nr = r[:]
                    nr[pos] = cand
                    if apply(nr):
                        replaced = True
                        improved = True
                        break
                if replaced:
                    break
            if improved:
                continue

        no_imp += 1

    return best


# ---------- Shake ----------

def shake(inst: Instance, sol: Solution, cons: int, post: int, t_end: float):
    """
    Perturbation (Shake):

    - entfernt cons aufeinanderfolgende Knoten ab Index post (zyklisch über Route)
    - baut eine Basisroute aus den verbleibenden Knoten
    - versucht, entfernte Knoten wieder stochastisch und budget-feasible zu re-insertieren

    Gibt neue Lösung und aktualisierte cons/post zurück.
    """
    global rng
    r = sol.route[:]
    if r[0] != r[-1]:
        r.append(r[0])
    m = len(r)
    if m <= 3:
        return sol, cons, post

    # Indizes der zu entfernenden Knoten bestimmen (innerhalb der Route)
    rem = []
    idx = max(1, min(post, m - 2))
    for _ in range(cons):
        rem.append(idx)
        idx += 1
        if idx >= m - 1:
            idx = 1
    rem_nodes = [r[i] for i in rem]
    for i in sorted(rem, reverse=True):
        r.pop(i)
    if r[0] != r[-1]:
        r.append(r[0])

    base = Solution(r, route_length(inst, r), route_score(inst, r))
    cur = base
    inserted = True

    # entfernte Knoten nacheinander wieder einfügen, solange möglich
    while inserted and rem_nodes and time.time() < t_end:
        if time.time() >= t_end:
            break
        inserted = False
        F = []
        for node in rem_nodes:
            if time.time() >= t_end:
                break
            for pos in range(len(cur.route) - 1):
                if time.time() >= t_end:
                    break
                feas, add = feasible_after_insert(inst, cur.route, pos, node, cur.length)
                if feas and add > 0:
                    ratio = (inst.scores[node] ** 2) / add
                    F.append((ratio, node, pos, add))
        if F:
            F.sort(key=lambda t: t[0], reverse=True)
            s = sum(x[0] for x in F)
            if s <= 0 or time.time() >= t_end:
                break
            u = rng.random() * s
            acc = 0.0
            chosen = F[-1]
            for it in F:
                acc += it[0]
                if u <= acc:
                    chosen = it
                    break
            _, node, pos, add = chosen
            nr = cur.route[:]
            nr.insert(pos + 1, node)
            cur = Solution(nr, route_length(inst, nr), route_score(inst, nr))
            rem_nodes.remove(node)
            inserted = True

    # Startposition der nächsten Shake-Region zyklisch weiterschieben
    post = post + cons
    if post >= len(cur.route) - 1:
        post = (post % (len(cur.route) - 1)) or 1
    return cur, cons, post


# ---------- ILS-Fassade ----------

def ttdp_ils(points: List[Tuple[float, float, float]], start: int, budget: float,
             seed: int = 0, time_limit_s: float = 5.0,
             f_keep: int = 30, threshold_intensify: int = 20, max_no_improve_ls: int = 30):
    """
    Komplettes ILS-Verfahren für eine TTDP-Instanz.

    Parameters
    ----------
    points : Liste von (x, y, score)
    start  : Index des Depots
    budget : maximale Routendistanz
    seed   : Random-Seed
    time_limit_s : Zeitbudget für den gesamten ILS-Lauf
    f_keep : Kandidatenlistenlänge bei Insertion
    threshold_intensify : nach dieser Anzahl nicht verbessernder Iterationen
                          wird aus der aktuellen Lösung wieder die Bestlösung gemacht
    max_no_improve_ls : Abbruchgrenze der Lokalsuche

    Returns
    -------
    route_nodes : Liste besuchter Knoten ohne abschließendes Depot
    best_score  : erreichte Score-Summe
    remaining   : verbleibendes Budget
    """
    global rng
    inst = Instance.from_points(points, budget, start)
    p = ILSParams(time_limit_s, seed, f_keep, threshold_intensify, max_no_improve_ls)
    rng = random.Random(seed)
    t_end = time.time() + time_limit_s

    # Startlösung + erste Lokalsuche
    cur = construct_initial(inst, p, t_end)
    cur = local_search(inst, cur, p, t_end)
    best = Solution(cur.route[:], cur.length, cur.score)

    no_imp = 0      # Anzahl Iterationen ohne Verbesserung
    cons = 1        # aktuelle Shake-Länge
    post = 1        # startender Index des zu shakenden Abschnitts
    fixed = 2       # nach 'fixed' Iterationen ohne Verbesserung Shake-Stärke erhöhen
    counter = 0

    while time.time() < t_end:
        # Shake + Lokalsuche
        cur, cons, post = shake(inst, cur, cons, post, t_end)
        cur = local_search(inst, cur, p, t_end)

        # Verbesserung der globalen Bestlösung?
        if cur.score > best.score + 1e-12 or (abs(cur.score - best.score) <= 1e-12 and cur.length < best.length - 1e-12):
            best = cur
            no_imp = 0
            cons = 1
            post = 1
            counter = 0
        else:
            no_imp += 1
            # Intensification: gelegentlich zur Bestlösung zurückspringen
            if (no_imp + 1) % p.threshold_intensify == 0:
                cur = Solution(best.route[:], best.length, best.score)
            counter += 1
            # Schrittweite der Perturbation langsam steigern
            if counter >= fixed:
                counter = 0
                cons += 1
            if cons > max(1, len(best.route) // 2):
                cons = 1
            if post >= len(best.route) - 1:
                post = 1

    # Route ohne abschließendes Depot zurückgeben
    rn = best.route[:-1] if best.route and best.route[0] == best.route[-1] else best.route[:]
    used = route_length(inst, best.route)
    rem = max(0.0, inst.budget - used)
    return rn, best.score, rem
