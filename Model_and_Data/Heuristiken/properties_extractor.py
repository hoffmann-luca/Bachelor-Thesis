"""Graph-Feature-Skelett für TTDP-Punktdateien"""
from typing import List, Tuple, Callable, Dict, Set
import math
from scipy.spatial import cKDTree as _KDTree
from scipy.spatial import Delaunay as _SciPyDelaunay
import numpy as np
from core import read_points as _read_points_core


def read_points(file_path: str) -> Tuple[List[Tuple[float, float, float]], float]:
    """
    Liest eine TTDP-Punktdatei.

    Falls core.read_points verfügbar ist, wird es verwendet.
    Format (wie in der restlichen BA):
      1–2: Header
      3:   irgendwas <budget>
      4+:  x y score

    Rückgabe:
        pts:   Liste (x, y, score)
        B:     Budget
    """
    if _read_points_core is not None:
        return _read_points_core(file_path)
    pts: List[Tuple[float, float, float]] = []
    B = 0.0
    k = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if k <= 1:
                k += 1
                continue
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
            if not line:
                k += 1
                continue
            p = line.split()
            if len(p) >= 3:
                pts.append((float(p[0]), float(p[1]), float(p[2])))
            k += 1
    return pts, B


# ---------------------------
# Datencontainer
# ---------------------------

class PointsDataset:
    """Leichter Container für Koordinaten, Scores und Budget einer Instanz."""
    def __init__(self, points: List[Tuple[float, float, float]], budget: float, start: int = 0):
        self.points = points
        self.budget = budget
        aelf = self  # keine Auswirkung auf Logik, historischer Style
        self.start = start
        self.coords = [(x, y) for (x, y, _) in points]
        self.scores = [s for (_, _, s) in points]
        self.n = len(points)

    @staticmethod
    def from_file(path: str, start: int = 0) -> "PointsDataset":
        """Erzeugt ein PointsDataset direkt aus einer TTDP-Textdatei."""
        pts, B = read_points(path)
        return PointsDataset(pts, B, start)


# ---------------------------
# Graphrepräsentation
# ---------------------------

class Graph:
    """
    Ungerichteter, einfacher Graph mit Adjazenz-Sets.

    - self.adj[u] ist die Menge der Nachbarn von u
    - Kanten werden immer symmetrisch hinzugefügt
    - Duplikate werden durch Set-Struktur automatisch vermieden
    """
    __slots__ = ("n", "adj", "_m")

    def __init__(self, n: int):
        self.n = int(n)
        self.adj = [set() for _ in range(self.n)]
        self._m = 0  # Anzahl ungerichteter Kanten

    def add_edge(self, u: int, v: int) -> None:
        """Fügt eine ungerichtete Kante {u,v} hinzu (ohne Schleifen & Duplikate)."""
        if u == v:
            return
        if v not in self.adj[u]:
            self.adj[u].add(v)
            self.adj[v].add(u)
            self._m += 1

    def degree(self, u: int) -> int:
        """Grad eines Knotens."""
        return len(self.adj[u])

    def degrees(self):
        """Liste der Grade aller Knoten."""
        return [len(nei) for nei in self.adj]

    def has_edge(self, u: int, v: int) -> bool:
        """True, falls {u,v} als Kante existiert."""
        return v in self.adj[u]

    def number_of_edges(self) -> int:
        """Anzahl ungerichteter Kanten."""
        return self._m

    def edges(self):
        """Iterator über alle Kanten (u < v)."""
        for u in range(self.n):
            for v in self.adj[u]:
                if v > u:
                    yield (u, v)

    def as_adj_lists(self):
        """Adjazenzlisten als normale Python-Listen (z.B. zum Debuggen)."""
        return [list(nei) for nei in self.adj]


# ---------------------------
# Distanz-Helfer
# ---------------------------

def euclid(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """Euklidische Distanz zwischen zwei 2D-Punkten."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


# ---------------------------
# Delaunay-Integration (SciPy/Qhull)
# ---------------------------

def _delaunay_tris(coords: List[Tuple[float, float]], qhull_options: str = "QJ") -> List[Tuple[int, int, int]]:
    """
    Berechnet die Delaunay-Triangulation der 2D-Koordinaten
    und gibt Dreiecke als Indextripel zurück.
    """
    tri = _SciPyDelaunay(coords, qhull_options=qhull_options)
    return [tuple(int(i) for i in t) for t in tri.simplices]


def _delaunay_edge_set_from_tris(tris: List[Tuple[int, int, int]]) -> Set[Tuple[int, int]]:
    """
    Leitet aus Delaunay-Dreiecken eine Menge ungerichteter Kanten ab.

    Jede Dreiecksseite (i,j), (j,k), (k,i) wird als sortiertes Tupel (min,max) gespeichert.
    """
    edges: Set[Tuple[int, int]] = set()
    for (i, j, k) in tris:
        a = (i, j) if i < j else (j, i)
        b = (j, k) if j < k else (k, j)
        c = (k, i) if k < i else (i, k)
        edges.add(a)
        edges.add(b)
        edges.add(c)
    return edges


# ---------------------------
# Graph-Builder
# ---------------------------

def build_complete_graph(ds: PointsDataset) -> Graph:
    """Kompletter Graph: jeder Knoten mit allen anderen verbunden."""
    g = Graph(ds.n)
    for i in range(ds.n):
        for j in range(i + 1, ds.n):
            g.add_edge(i, j)
    return g


def build_radius_graph(ds: PointsDataset, radius: float) -> Graph:
    """
    Radius-Graph: Kante zwischen u und v, wenn Distanz(u,v) <= radius.

    Nachbarn werden per KD-Tree gesucht, es werden nur Paare mit v > u eingetragen,
    um doppelte Kanten zu vermeiden.
    """
    from math import isfinite
    if not isfinite(radius) or radius <= 0:
        raise ValueError(f"radius muss > 0 sein, bekommen: {radius}")

    tree = _KDTree(ds.coords)
    g = Graph(ds.n)

    for u in range(ds.n):
        nbrs = tree.query_ball_point(ds.coords[u], r=radius)
        for v in nbrs:
            if v == u:
                continue
            if v > u:
                g.add_edge(u, v)
    return g


def build_knn_graph(ds: PointsDataset, k: int) -> Graph:
    """
    k-NN-Graph (ungerichtet): jeder Punkt verbindet sich mit seinen k nächsten Nachbarn.
    Die gerichteten Kanten werden symmetriert (nur einmal pro Paar).
    """
    tree = _KDTree(ds.coords)
    # dists: (n, k+1), idx: (n, k+1) – erste Spalte ist jeweils self
    dists, idx = tree.query(ds.coords, k=k + 1)
    g = Graph(ds.n)

    for u in range(ds.n):
        for v in idx[u][1:]:  # self überspringen
            if v < 0 or v == u:
                continue
            a, b = (u, int(v)) if u < v else (int(v), u)
            if a != b:
                g.add_edge(a, b)
    return g


def build_delaunay_graph(ds: PointsDataset, qhull_options: str = "QJ", precomputed_edges=None) -> Graph:
    """
    Delaunay-Graph: Kanten sind die Kanten der Delaunay-Triangulation.

    precomputed_edges erlaubt das Wiederverwenden bereits berechneter Delaunay-Kanten.
    """
    edges = precomputed_edges if precomputed_edges is not None else _delaunay_edge_set_from_tris(_delaunay_tris(ds.coords))
    g = Graph(ds.n)
    for u, v in edges:
        g.add_edge(u, v)
    return g


class _DSU(object):
    """Disjoint-Set-Union (Union-Find) für Kruskal auf Delaunay-Kanten."""
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1
        return True


def build_mst_via_delaunay(ds: PointsDataset, qhull_options: str = "QJ", precomputed_edges=None) -> Graph:
    """
    Euklidischer MST über Delaunay-Kanten (Kruskal).

    Idee: MST-Kantenmenge ist Teil der Delaunay-Triangulation, daher müssen
    nur diese Kanten sortiert und per Union-Find ausgewählt werden.
    """
    edges = precomputed_edges if precomputed_edges is not None else _delaunay_edge_set_from_tris(_delaunay_tris(ds.coords))
    E = [(euclid(ds.coords[u], ds.coords[v]), u, v) for (u, v) in edges]
    E.sort(key=lambda t: t[0])
    g = Graph(ds.n)
    dsu = _DSU(ds.n)
    added = 0
    for _, u, v in E:
        if dsu.union(u, v):
            g.add_edge(u, v)
            added += 1
            if added == ds.n - 1:
                break
    return g


def build_nng_via_delaunay(ds: PointsDataset, qhull_options: str = "QJ", mutual: bool = False, precomputed_edges=None) -> Graph:
    """
    Nearest-Neighbor-Graph (aus Delaunay-Kanten).

    - mutual=False: alle NN-Paare, symmetrisiert (jede undirektionale Kante einmal)
    - mutual=True: nur gegenseitige NN-Paare (u ist NN von v UND v ist NN von u)
    """
    edges = precomputed_edges if precomputed_edges is not None else _delaunay_edge_set_from_tris(_delaunay_tris(ds.coords))
    neigh = [set() for _ in range(ds.n)]
    for u, v in edges:
        neigh[u].add(v)
        neigh[v].add(u)

    nn = [-1] * ds.n
    for u in range(ds.n):
        if not neigh[u]:
            continue
        uu = ds.coords[u]
        nn[u] = min(neigh[u], key=lambda v: euclid(uu, ds.coords[v]))

    g = Graph(ds.n)
    if mutual:
        for u, v in enumerate(nn):
            if v != -1 and nn[v] == u and u < v:
                g.add_edge(u, v)
    else:
        seen = set()
        for u, v in enumerate(nn):
            if v == -1:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in seen:
                seen.add((a, b))
                g.add_edge(a, b)
    return g


# ---------------------------
# Feature-API
# ---------------------------

FeatureFn = Callable[[PointsDataset, Graph], Dict[str, float]]


def feature_basic_counts(ds: PointsDataset, g: Graph) -> Dict[str, float]:
    """
    Basis-Features:

    - nodes:      Anzahl Knoten
    - degree_min: minimaler Grad
    - degree_max: maximaler Grad
    - degree_avg: mittlerer Grad
    """
    degs = g.degrees()
    n = float(ds.n)
    if degs:
        s = sum(degs)
        return {
            "nodes": n,
            "degree_min": float(min(degs)),
            "degree_max": float(max(degs)),
            "degree_avg": float(s / len(degs)),
        }
    else:
        return {"nodes": n, "degree_min": 0.0, "degree_max": 0.0, "degree_avg": 0.0}


def feature_connected_components(ds: PointsDataset, g: Graph) -> Dict[str, float]:
    """
    Zusammenhangskomponenten des Graphen:

    - components:        Anzahl Komponenten
    - component_largest: Größe größte Komponente
    - component_avg_size: Durchschnittsgröße
    """
    n = g.n
    seen = [False] * n
    sizes: List[int] = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        cnt = 0
        while stack:
            v = stack.pop()
            cnt += 1
            for w in g.adj[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        sizes.append(cnt)
    if not sizes:
        return {"components": 0.0}
    comps = len(sizes)
    largest = max(sizes)
    avg = sum(sizes) / float(comps)
    return {
        "components": float(comps),
        "component_largest": float(largest),
        "component_avg_size": float(avg),
    }


def feature_clustering(ds: PointsDataset, g: Graph) -> Dict[str, float]:
    """
    Globaler Clustering-Koeffizient:

    Mittelwert des lokalen Clustering-Koeffizienten über alle Knoten mit Grad ≥ 2.
    """
    n = g.n
    if n == 0:
        return {"clustering_avg": 0.0}
    neigh_sets = [set(nei) for nei in g.adj]
    total = 0.0
    cnt = 0
    for v in range(n):
        d = len(neigh_sets[v])
        if d < 2:
            continue
        Nv = list(neigh_sets[v])
        t = 0
        for i in range(d):
            u = Nv[i]
            Nu = neigh_sets[u]
            for j in range(i + 1, d):
                w = Nv[j]
                if w in Nu:
                    t += 1
        Cv = (2.0 * t) / (d * (d - 1))
        total += Cv
        cnt += 1
    return {"clustering_avg": (total / float(cnt)) if cnt > 0 else 0.0}


def feature_hull_stats(ds, g, qhull_options: str = "QJ"):
    """
    Kennzahlen der konvexen Hülle (SciPy/Qhull):

    - hull_area:        Fläche
    - hull_perimeter:   Umfang
    - hull_vertices:    Anzahl Hüllpunkte
    - hull_compactness: kompakte Form (≈1 für kreisähnlich)
    - density:          Knotendichte (n / Fläche)

    Fällt SciPy weg oder ist die Hülle degeneriert, werden sinnvolle Defaults geliefert.
    """
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return {
            "hull_area": 0.0,
            "hull_perimeter": 0.0,
            "hull_vertices": float(ds.n if ds.n < 3 else 0),
            "hull_compactness": 0.0,
            "density": 0.0,
        }

    if ds.n < 3:
        return {
            "hull_area": 0.0,
            "hull_perimeter": 0.0,
            "hull_vertices": float(ds.n),
            "hull_compactness": 0.0,
            "density": 0.0,
        }

    hull = ConvexHull(ds.coords, qhull_options=qhull_options)

    hv = hull.vertices
    perim = 0.0
    for i in range(len(hv)):
        a = ds.coords[hv[i]]
        b = ds.coords[hv[(i + 1) % len(hv)]]
        perim += math.hypot(a[0] - b[0], a[1] - b[1])

    area = float(getattr(hull, "volume", 0.0))
    if area == 0.0 and hasattr(hull, "area"):
        area = float(hull.area)

    compact = (4.0 * math.pi * (area if area > 0 else 1.0)) / (
        perim * perim if perim > 0 else 1.0
    )
    density = ds.n / (area if area > 0 else 1.0)

    return {
        "hull_area": area,
        "hull_perimeter": perim,
        "hull_vertices": float(len(hv)),
        "hull_compactness": compact,
        "density": density,
    }


def _edge_lengths(ds, g):
    """Iterator über alle Kantenlängen (jede ungerichtete Kante genau einmal)."""
    for u in range(g.n):
        pu = ds.coords[u]
        for v in g.adj[u]:
            if v > u:
                yield euclid(pu, ds.coords[v])


def feature_mst_stats(ds, g):
    """
    MST-Deskriptoren:

    - mst_edge_sum: Gesamtlänge
    - mst_edge_max: längste Kante
    - mst_leaves:   Anzahl Blätter
    - mst_diameter: Durchmesser bzgl. Kantenlängen
    """
    degs = g.degrees()
    lengths = list(_edge_lengths(ds, g))
    total = sum(lengths)
    mx = max(lengths) if lengths else 0.0
    leaves = sum(1 for d in degs if d == 1)

    import heapq

    def dijkstra(start):
        dist = [math.inf] * g.n
        dist[start] = 0.0
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v in g.adj[u]:
                w = euclid(ds.coords[u], ds.coords[v])
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        far = max(range(g.n), key=lambda i: dist[i])
        return far, dist

    if g.n > 0 and any(g.adj):
        a, _ = dijkstra(0)
        b, dist_a = dijkstra(a)
        diameter = dist_a[b]
    else:
        diameter = 0.0

    return {
        "mst_edge_sum": float(total),
        "mst_edge_max": float(mx),
        "mst_leaves": float(leaves),
        "mst_diameter": float(diameter),
    }


def _mst_edge_lengths(ds, g):
    """Iterator über ungerichtete MST-Kantenlängen."""
    for u in range(g.n):
        xu, yu = ds.coords[u]
        for v in g.adj[u]:
            if v > u:
                xv, yv = ds.coords[v]
                yield math.hypot(xu - xv, yu - yv)


def _hull_area(ds, qhull_options="QJ"):
    """Nur die Hüllfläche (für Normierungen). Gibt 0 oder None bei Fehlern zurück."""
    try:
        from scipy.spatial import ConvexHull
        if ds.n < 3:
            return 0.0
        hull = ConvexHull(ds.coords, qhull_options=qhull_options)
        area = float(getattr(hull, "volume", 0.0) or getattr(hull, "area", 0.0))
        return area if area > 0 else None
    except Exception:
        return None


def _median_nn_distance(ds):
    """
    Median der Nächster-Nachbar-Distanzen (über KD-Tree).

    Liefert einen robusten Skalenparameter m für die Punktdichte.
    """
    coords = np.asarray(ds.coords)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return float("nan")

    tree = _KDTree(coords)
    dists, _ = tree.query(coords, k=2)  # self + echter NN
    nn = dists[:, 1]
    return float(np.median(nn))


def feature_mst_norms_both(ds, g, qhull_options: str = "QJ"):
    """
    Dimensionslose MST-Normierungen:

      - mst_edge_sum_per_sqrtA = Summe(MST-Kanten) / √Area
      - mst_edge_max_per_sqrtA = max(MST-Kante) / √Area
      - mst_edge_sum_units_s   = Summe(MST-Kanten) / ((n-1)*s),  s = √(Area/n)
      - mst_edge_sum_units_m   = Summe(MST-Kanten) / ((n-1)*m),  m = Median(NN)
    """
    edges = list(_mst_edge_lengths(ds, g))
    if not edges:
        return {
            "mst_edge_sum_per_sqrtA": 0.0,
            "mst_edge_max_per_sqrtA": 0.0,
            "mst_edge_sum_units_s": 0.0,
            "mst_edge_sum_units_m": 0.0,
        }

    total = float(sum(edges))
    emax = float(max(edges))
    A = _hull_area(ds, qhull_options=qhull_options)

    if A and A > 0.0:
        sqrtA = math.sqrt(A)
        sum_per_sqrtA = total / sqrtA
        max_per_sqrtA = emax / sqrtA
        s = math.sqrt(A / max(1, ds.n))
        sum_units_s = total / (max(1, ds.n - 1) * s)
    else:
        sum_per_sqrtA = 0.0
        max_per_sqrtA = 0.0
        sum_units_s = 0.0

    m = _median_nn_distance(ds)
    sum_units_m = total / (max(1, ds.n - 1) * (m if m > 0 else 1.0))

    return {
        "mst_edge_sum_per_sqrtA": float(sum_per_sqrtA),
        "mst_edge_max_per_sqrtA": float(max_per_sqrtA),
        "mst_edge_sum_units_s": float(sum_units_s),
        "mst_edge_sum_units_m": float(sum_units_m),
    }


# ---------------------------
# Pipeline
# ---------------------------

class FeaturePipeline:
    """
    Pipeline zum Berechnen von Features:

    - builder: Graph-Builder-Funktion (nimmt PointsDataset, **kwargs)
    - features: Liste von Featurefunktionen (ds, g) -> Dict
    """
    def __init__(self, builder: Callable[..., Graph], *features: FeatureFn, **builder_kwargs):
        self.builder = builder
        self.features = list(features)
        self.builder_kwargs = builder_kwargs

    def run(self, ds: PointsDataset) -> Dict[str, float]:
        """Erzeugt einen Graphen und vereinigt alle Feature-Dicts in einem Ergebnis-Dict."""
        g = self.builder(ds, **self.builder_kwargs)
        out: Dict[str, float] = {}
        for f in self.features:
            out.update(f(ds, g))
        return out


# ---------------------------
# Beispiel-Nutzung per CLI
# ---------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Graph-Feature-Skelett (mit SciPy-Delaunay)")
    ap.add_argument("file")
    ap.add_argument(
        "--builder",
        choices=["complete", "radius", "knn", "delaunay", "mst_delaunay"],
        default="complete",
    )
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--qhull-options",
        type=str,
        default="QJ",
        help="Optionen für Qhull (SciPy Delaunay), z. B. '', 'Qbb Qc', default 'QJ'.",
    )
    args = ap.parse_args()

    ds = PointsDataset.from_file(args.file)

    if args.builder == "complete":
        pipe = FeaturePipeline(
            build_complete_graph,
            feature_basic_counts,
            feature_clustering,
            feature_connected_components,
        )
    elif args.builder == "radius":
        pipe = FeaturePipeline(
            build_radius_graph,
            feature_basic_counts,
            radius=args.radius,
        )
    elif args.builder == "knn":
        pipe = FeaturePipeline(
            build_knn_graph,
            feature_basic_counts,
            k=args.k,
        )
    elif args.builder == "delaunay":
        pipe = FeaturePipeline(
            build_delaunay_graph,
            feature_basic_counts,
            feature_connected_components,
            qhull_options=args.qhull_options,
        )
    else:  # mst_delaunay
        pipe = FeaturePipeline(
            build_mst_via_delaunay,
            feature_basic_counts,
            feature_connected_components,
            qhull_options=args.qhull_options,
        )

    feats = pipe.run(ds)
    for k, v in feats.items():
        print("{}:\t{}".format(k, v))
