from typing import List, Tuple
import random, time
from core import Instance, route_length, route_score, _normalize_route

EPS = 1e-12
rng = None


class GRASPParams:
    """Parametercontainer für GRASP-SR."""
    def __init__(self, alpha=0.2, time_limit_s=5.0, seed=0, max_cons_mult=2, max_ls_rounds=3):
        self.alpha = alpha              # RCL-Steuerung in der Konstruktion
        self.time_limit_s = time_limit_s
        self.seed = seed
        self.max_cons_mult = max_cons_mult   # Faktor für max. Konstruktionsschritte
        self.max_ls_rounds = max_ls_rounds   # max. Runden Lokalsuche pro Iteration


# --------- lokale Verbesserer ---------

def _two_opt_reduce(inst: Instance, route: List[int], t_end: float = None) -> List[int]:
    """
    2-opt Reduktion (Depot fixiert), bricht ab, wenn t_end überschritten ist.

    Route wird zwischendurch immer wieder normalisiert, damit keine
    überflüssigen Depot-Wiederholungen auftreten.
    """
    r = _normalize_route(route[:], inst.start)
    improved = True
    while improved:
        if t_end is not None and time.time() >= t_end:
            return r
        improved = False
        m = len(r)
        for i in range(1, m - 2):
            if t_end is not None and time.time() >= t_end:
                return r
            for j in range(i + 1, m - 1):
                if t_end is not None and time.time() >= t_end:
                    return r
                a, b = r[i - 1], r[i]
                c, d = r[j], r[(j + 1) % m]
                delta = inst.dist[a][c] + inst.dist[b][d] - inst.dist[a][b] - inst.dist[c][d]
                if delta < -EPS:
                    r = r[:i] + list(reversed(r[i:j + 1])) + r[j + 1:]
                    r = _normalize_route(r, inst.start)
                    improved = True
                    break
            if improved:
                break
    return r


def _best_insert_pos(inst: Instance, route: List[int], node: int) -> Tuple[int, float]:
    """
    Bestimmt die Position in der Route, an der node die geringste Längenveränderung erzeugt.

    Rückgabe:
        (pos, delta_len) mit pos = Insertionsposition (nach pos wird eingefügt)
    """
    r = _normalize_route(route, inst.start)
    best_pos, best_add = 1, float("inf")
    for pos in range(len(r) - 1):
        a, b = r[pos], r[(pos + 1) % len(r)]
        add = inst.dist[a][node] + inst.dist[node][b] - inst.dist[a][b]
        if add < best_add:
            best_add, best_pos = add, pos
    return best_pos, best_add


# --------- Konstruktionsschritt ---------

def _add_vertex(
    inst: Instance,
    cur_path: List[int],
    cur_len: float,
    cur_score: float,
    alpha: float,
    blocked: int = -1,
    t_end: float = None,
):
    """
    Versucht, einen weiteren Knoten in die aktuelle Route einzufügen.

    Zwei Modi:
      1) Fall T <= Budget: reine Insert-Operation
      2) Fall T > Budget: Insert + Entfernen eines Segments (kleine Route-Operation)

    Es wird eine Kandidatenliste (CL) aufgebaut; aus der RCL (filtered via alpha) wird
    zufällig ein Kandidat gewählt und noch mit 2-opt reduziert.

    Rückgabe:
        (added?, new_path, new_len, new_score)
    """
    n = len(inst.coords)
    r = _normalize_route(cur_path, inst.start)
    CL = []  # Einträge: (route, length, score, gain)
    in_r = set(r)
    in_r.discard(inst.start)

    for w in range(n):
        if t_end is not None and time.time() >= t_end:
            break
        if w == inst.start or w in in_r or w == blocked:
            continue

        pos, add = _best_insert_pos(inst, r, w)
        rp = r[:]
        rp.insert(pos + 1, w)
        rp = _normalize_route(rp, inst.start)
        T = route_length(inst, rp)

        if T <= inst.budget + EPS:
            # Insert bleibt im Budget: klassischer Kandidat
            P = route_score(inst, rp)
            gain = P - cur_score
            if gain > EPS or (abs(gain) <= EPS and T < cur_len - EPS):
                CL.append((rp, T, P, gain))
        else:
            # Insert verletzt Budget: zusätzliches Entfernen eines Segments prüfen
            if t_end is not None and time.time() >= t_end:
                break
            best_rr = None
            rin = rp
            m = len(rin)
            for i in range(1, m - 1):
                if t_end is not None and time.time() >= t_end:
                    break
                for j in range(i, m - 2):
                    if t_end is not None and time.time() >= t_end:
                        break
                    rr = rin[:i] + rin[j + 1:]
                    rr = _normalize_route(rr, inst.start)
                    T2 = route_length(inst, rr)
                    if T2 <= inst.budget + EPS:
                        P2 = route_score(inst, rr)
                        gain2 = P2 - cur_score
                        if gain2 > EPS or (abs(gain2) <= EPS and T2 < cur_len - EPS):
                            best_rr = (rr, T2, P2, gain2)
                        # erstes zulässiges Segment reicht hier als Kandidat
                        break
            if best_rr is not None:
                CL.append(best_rr)

    if not CL or (t_end is not None and time.time() >= t_end):
        return False, r, cur_len, cur_score

    # RCL: nach Verhältnis gain / effektive Mehrkosten filtern
    rmax = max(g / max(t - cur_len, EPS) for (_, t, _, g) in CL)
    thresh = rmax * alpha
    RCL = [t for t in CL if (t[3] / max(t[1] - cur_len, EPS)) >= thresh - EPS]

    # zufällige Wahl aus RCL und 2-opt-Verbesserung
    global rng
    rp, tl, ps, _ = rng.choice(RCL)
    rp = _two_opt_reduce(inst, rp, t_end=t_end)
    tl = route_length(inst, rp)
    ps = route_score(inst, rp)

    # Schrittannahme nur bei Verbesserung (Score↑ oder Score= & L↓)
    if (ps > cur_score + EPS) or (abs(ps - cur_score) <= EPS and tl < cur_len - EPS):
        return True, rp, tl, ps
    else:
        return False, r, cur_len, cur_score


# --------- Lokalsuche ---------

def _localsearch(
    inst: Instance,
    best_path: List[int],
    best_len: float,
    best_score: float,
    alpha: float,
    t_end: float,
):
    """
    Einfache Lokalsuche:

    - für jeden inneren Knoten w:
        * w entfernen, Basisroute mit 2-opt verbessern
        * danach wieder greedy/GRASP-basiert auffüllen (Add-Versuche mit _add_vertex)
    """
    improved = False
    r = _normalize_route(best_path, inst.start)
    inner = [v for v in r[1:-1] if v != inst.start]

    for w in inner:
        if time.time() >= t_end:
            break

        # w entfernen -> Basisroute
        base = [x for x in r if x != w]
        base = _two_opt_reduce(inst, base, t_end=t_end)
        T = route_length(inst, base)
        P = route_score(inst, base)

        cur_added = True
        cur_path, cur_len, cur_score = base, T, P
        add_steps = 0
        add_steps_cap = 2 * len(inst.coords)

        # begrenzte Add-Versuche mit w als "gesperrtem" Knoten
        while cur_added and time.time() < t_end and add_steps < add_steps_cap:
            cur_added, cur_path, cur_len, cur_score = _add_vertex(
                inst, cur_path, cur_len, cur_score, alpha, blocked=w, t_end=t_end
            )
            add_steps += 1

        if (cur_score > best_score + EPS) or (
            abs(cur_score - best_score) <= EPS and cur_len < best_len - EPS
        ):
            best_path, best_len, best_score = cur_path, cur_len, cur_score
            improved = True

    return improved, best_path, best_len, best_score


# --------- GRASP-Fassade ---------

def ttdp_grasp_sr(
    points: List[Tuple[float, float, float]],
    start: int,
    budget: float,
    alpha: float = 0.2,
    seed: int = 0,
    time_limit_s: float = 5.0,
):
    """
    GRASP-SR für TTDP.

    Eingabe:
        points : Liste (x, y, score)
        start  : Depotindex
        budget : Maximale Routendistanz
        alpha  : GRASP-Parameter für RCL
        seed   : RNG-Seed
        time_limit_s : harte Zeitgrenze für die gesamte Heuristik

    Rückgabe:
        (route_nodes, best_score, remaining_budget)
        route_nodes enthält die besuchten Knoten ohne abschließende Depot-Wiederholung.
    """
    inst = Instance.from_points(points, budget, start)
    params = GRASPParams(alpha, time_limit_s, seed)
    global rng
    rng = random.Random(seed)
    t_end = time.time() + params.time_limit_s

    best_route = [start, start]
    best_len = 0.0
    best_score = 0.0

    # Mehrere GRASP-Iterationen, solange noch Zeit ist
    while time.time() < t_end:
        # Construction
        cur_path = [start, start]
        cur_len = 0.0
        cur_score = 0.0
        added = True
        cons_steps = 0
        cons_cap = params.max_cons_mult * max(1, len(inst.coords))

        while added and time.time() < t_end and cons_steps < cons_cap:
            added, cur_path, cur_len, cur_score = _add_vertex(
                inst, cur_path, cur_len, cur_score, params.alpha, t_end=t_end
            )
            cons_steps += 1

        # Normalisieren + Bewertung der konstruierten Route
        cur_path = _normalize_route(cur_path, start)
        cur_len = route_length(inst, cur_path)
        cur_score = route_score(inst, cur_path)

        # Local Search mit begrenzter Rundenzahl
        ls_round = 0
        improved = True
        while improved and time.time() < t_end and ls_round < params.max_ls_rounds:
            improved, cur_path, cur_len, cur_score = _localsearch(
                inst, cur_path, cur_len, cur_score, params.alpha, t_end
            )
            cur_path = _normalize_route(cur_path, start)
            cur_len = route_length(inst, cur_path)
            cur_score = route_score(inst, cur_path)
            ls_round += 1

        # Bestlösung aktualisieren
        if (cur_score > best_score + EPS) or (
            abs(cur_score - best_score) <= EPS and cur_len < best_len - EPS
        ):
            best_route, best_len, best_score = cur_path, cur_len, cur_score

    # Route ohne abschließenden Depot-Knoten zurückgeben
    route_nodes = (
        best_route[:-1] if best_route and best_route[0] == best_route[-1] else best_route[:]
    )
    rem = max(0.0, inst.budget - route_length(inst, best_route))
    return route_nodes, best_score, rem
