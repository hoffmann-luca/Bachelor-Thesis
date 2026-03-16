"""
Hilfsfunktionen zum Plotten von TSP/Orienteering-Touren mit Matplotlib.

Kernidee:
- points: vollständige Punktmenge (Liste von (x, y, score))
- paths:   eine Route oder mehrere Routen (als Indices oder Koordinaten)
- otherPoints: Punkte, die nicht in der Tour liegen (z.B. zur Visualisierung von Restknoten)

Funktion plotTSP:
- normalisiert paths (Indices vs. Koordinaten)
- skaliert und jittert die Punkte für eine schönere Darstellung
- zeichnet Tour(en) mit Pfeilen und optionalen Labels
- markiert ungenutzte Punkte separat
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import random
from matplotlib.patches import FancyArrowPatch


def _text_color_for_marker(matplotlib_color):
    """
    Wählt Schwarz/Weiß als Textfarbe abhängig von der Helligkeit
    der Markerfarbe, damit Labels gut lesbar bleiben.
    """
    try:
        r, g, b = mcolors.to_rgb(matplotlib_color)
    except Exception:
        return 'black'
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < 0.5 else 'black'


def _apply_expand_and_jitter(coords, expand=1.0, jitter=0.0, seed=None):
    """
    Skaliert das Koordinatensystem um den Schwerpunkt und fügt optional
    zufälliges Jittern hinzu, um Überlagerungen zu reduzieren.

    - expand > 1: Punkte werden vom Schwerpunkt weggezogen
    - jitter > 0: zufällige kleine Offsets in x/y
    """
    if seed is not None:
        random.seed(seed)
    if not coords or (expand == 1.0 and (not jitter or jitter == 0.0)):
        return coords[:]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    out = []
    for x, y in coords:
        nx = cx + (x - cx) * expand
        ny = cy + (y - cy) * expand
        if jitter and jitter > 0:
            nx += random.uniform(-jitter, jitter)
            ny += random.uniform(-jitter, jitter)
        out.append((nx, ny))
    return out


def plotTSP(paths, otherPoints, points, num_iters=1,
            annotate=True, annotate_only='tour',
            label_fontsize=7, label_bold=False,
            expand=1.35, jitter=0.005, seed=None,
            used_size=200, unused_size=200,
            # Pfeil-/Linienparameter
            arrow_linewidth=0.7, arrow_alpha=0.9, arrow_head_scale=6.0, arrow_style='-|>'):
    """
    Plot für TSP/Orienteering mit Pfeilen und Labels.

    Wichtige Parameter:
    - paths:
        * entweder eine Route (Liste von Indices oder Koordinaten)
        * oder Liste von Routen (bei num_iters > 1)
    - otherPoints:
        * weitere Punkte, die nicht auf der Route liegen (Indices oder Koordinaten)
    - points:
        * vollständige Punkteliste (x, y, score), von der die Indices stammen
    - expand, jitter:
        * optische Anpassung der Koordinaten (Skalierung + zufälliges Jittern)
    - annotate:
        * True → Punkte werden beschriftet
    - annotate_only:
        * 'tour' → nur Tourpunkte beschriften
        * 'all'  → alle Punkte beschriften
    - arrow_*:
        * feine Kontrolle über das Erscheinungsbild der Pfeile
    """

    # --- Hilfsfunktionen, um zwischen Indices und Koordinaten zu mappen ---

    def coord_of_index(idx):
        p = points[idx]
        return (p[0], p[1])

    def find_index_by_coord(coord, tolerance=1e-6):
        """Sucht zu einer (x,y)-Koordinate den passenden Index in 'points'."""
        x, y = coord
        for idx, p in enumerate(points):
            px, py = p[0], p[1]
            if abs(px - x) <= tolerance and abs(py - y) <= tolerance:
                return idx
        return None

    # Farben für Plot-Elemente
    used_color = '#d62728'      # rot (benutzte Tourpunkte)
    unused_color = '#87CEFA'    # hellblau (nicht benutzte Punkte)
    background_color = 'lightgray'

    # Pfade normalisieren:
    # - paths kann eine einzelne Route oder Liste von Routen sein
    if num_iters > 1 and isinstance(paths, list) and len(paths) == num_iters and any(isinstance(el, list) for el in paths):
        path_list = paths
    else:
        path_list = [paths]

    normalized_paths = []
    for path in path_list:
        if not path:
            normalized_paths.append([])
            continue
        if all(isinstance(el, int) for el in path):
            # bereits als Indexliste gegeben
            normalized_paths.append(list(path))
        else:
            # Koordinaten → Indices versuchen, sonst als 'coord'-Marker behalten
            mapped = []
            for coord in path:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    idx = find_index_by_coord(coord)
                    if idx is None:
                        mapped.append(('coord', (coord[0], coord[1])))
                    else:
                        mapped.append(idx)
            normalized_paths.append(mapped)

    # otherPoints in Index/Koordinaten-Sicht aufspalten
    other_coords = []
    other_indices = []
    for op in (otherPoints or []):
        if isinstance(op, int):
            other_indices.append(op)
            other_coords.append(coord_of_index(op))
        elif isinstance(op, (list, tuple)) and len(op) >= 2:
            idx = find_index_by_coord(op)
            other_indices.append(idx)
            other_coords.append((op[0], op[1]))

    # Alle relevanten Koordinaten sammeln, für expand/jitter vorbereiten
    unique_coords = []
    unique_keys = []
    # Tourpunkte
    for path in normalized_paths:
        for el in path:
            if isinstance(el, int):
                key = ('i', el)
                if key not in unique_keys:
                    unique_keys.append(key)
                    unique_coords.append(coord_of_index(el))
            elif isinstance(el, tuple) and el and el[0] == 'coord':
                key = ('c', el[1])
                if key not in unique_keys:
                    unique_keys.append(key)
                    unique_coords.append(el[1])
    # weitere (nicht benutzte) Punkte
    for idx, coord in zip(other_indices, other_coords):
        key = ('i', idx) if idx is not None else ('c', coord)
        if key not in unique_keys:
            unique_keys.append(key)
            unique_coords.append(coord)
    # optional: auch alle Hintergrundpunkte berücksichtigen
    try:
        bg_coords = [(p[0], p[1]) for p in points]
        for c in bg_coords:
            key = ('c', c)
            if key not in unique_keys:
                unique_keys.append(key)
                unique_coords.append(c)
    except Exception:
        pass

    # expand + jitter anwenden und Mapping (Key → Anzeige-Koordinate) bauen
    display_coords = _apply_expand_and_jitter(unique_coords, expand=expand, jitter=jitter, seed=seed)
    display_points_map = {k: dc for k, dc in zip(unique_keys, display_coords)}

    def display_coord_for(el):
        """
        Liefert die gejitterte/expandierte Koordinate für:
        - Index (int)
        - ('coord', (x,y))-Marker
        """
        if isinstance(el, int):
            return display_points_map.get(('i', el), coord_of_index(el))
        elif isinstance(el, tuple) and el and el[0] == 'coord':
            return display_points_map.get(('c', el[1]), el[1])
        else:
            return None

    # Plot-Setup
    plt.figure(figsize=(10, 10))
    all_x = []
    all_y = []

    # Hintergrundpunkte leicht einzeichnen (alle Punkte aus 'points')
    try:
        bg_disp = []
        for p in points:
            key = ('c', (p[0], p[1]))
            dc = display_points_map.get(key, (p[0], p[1]))
            bg_disp.append(dc)
        bg_x = [c[0] for c in bg_disp]
        bg_y = [c[1] for c in bg_disp]
        plt.scatter(bg_x, bg_y, s=8, alpha=0.12, color=background_color, zorder=1)
        all_x.extend(bg_x)
        all_y.extend(bg_y)
    except Exception:
        pass

    # Tour(en) zeichnen
    for pi, path in enumerate(normalized_paths):
        ds = [display_coord_for(el) for el in path]
        ds = [d for d in ds if d is not None]
        if not ds:
            continue
        xs = [d[0] for d in ds]
        ys = [d[1] for d in ds]

        # Tourpunkte (Marker)
        plt.scatter(xs, ys, s=used_size, c=used_color, edgecolors='k', linewidths=0.9, zorder=6)

        # Tourkanten: dünne Linie + einzelne Pfeilspitzen
        if len(xs) > 1:
            # geschlossene Linie
            plt.plot(xs + [xs[0]], ys + [ys[0]], linewidth=arrow_linewidth,
                     color=used_color, alpha=arrow_alpha, zorder=5)

            # Pfeilspitzen pro Segment
            for i in range(len(xs)):
                x0, y0 = xs[i], ys[i]
                x1, y1 = xs[(i + 1) % len(xs)], ys[(i + 1) % len(xs)]
                dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                # Skalierung des Pfeilkopfs abhängig von Segmentlänge (aber begrenzt)
                mutation = max(2.0, arrow_head_scale * (dist if dist > 0 else 1.0) ** 0.5)
                mutation = min(mutation, 12.0)
                # Start/Ende leicht nach innen versetzen, damit Pfeil nicht direkt im Markerzentrum startet
                shrink = 0.15
                sx = x0 + (x1 - x0) * shrink
                sy = y0 + (y1 - y0) * shrink
                ex = x1 - (x1 - x0) * shrink
                ey = y1 - (y1 - y0) * shrink
                arr = FancyArrowPatch(
                    (sx, sy), (ex, ey),
                    arrowstyle=arrow_style,
                    mutation_scale=mutation,
                    linewidth=0.0,  # Linie bereits gezeichnet; hier nur Kopf
                    color=used_color,
                    alpha=arrow_alpha,
                    zorder=7,
                )
                plt.gca().add_patch(arr)

        all_x.extend(xs)
        all_y.extend(ys)

        # Labels (Tourpunkte oder alle, abhängig von annotate_only)
        if annotate and annotate_only in ('tour', 'all'):
            base_txt_color = _text_color_for_marker(used_color)
            stroke_color = 'black' if base_txt_color == 'white' else 'white'
            pe_effects = [pe.withStroke(linewidth=4.0, foreground=stroke_color), pe.Normal()]
            for el in path:
                dcoord = display_coord_for(el)
                if dcoord is None:
                    continue
                xdisp, ydisp = dcoord
                lbl = str(el) if isinstance(el, int) else f"({xdisp:.2f},{ydisp:.2f})"
                plt.text(
                    xdisp, ydisp, lbl,
                    fontsize=label_fontsize,
                    fontweight='bold' if label_bold else 'normal',
                    ha='center', va='center',
                    color=base_txt_color,
                    path_effects=pe_effects,
                    zorder=8,
                )

    # Unbenutzte Punkte deutlicher (hellblau) darstellen
    if other_coords:
        ox = []
        oy = []
        for idx, orig_coord in zip(other_indices, other_coords):
            if idx is not None:
                xdisp, ydisp = display_coord_for(idx)
            else:
                key = ('c', orig_coord)
                xdisp, ydisp = display_points_map.get(key, orig_coord)
            ox.append(xdisp)
            oy.append(ydisp)
            plt.scatter([xdisp], [ydisp], s=unused_size, c=unused_color,
                        marker='o', edgecolors='k', linewidths=0.6, zorder=4)
            if annotate and annotate_only == 'all':
                base_txt_color2 = 'black'
                stroke_color2 = 'white'
                pe_effects2 = [pe.withStroke(linewidth=3.0, foreground=stroke_color2), pe.Normal()]
                lbl = str(idx) if idx is not None else f"({orig_coord[0]:.2f},{orig_coord[1]:.2f})"
                plt.text(
                    xdisp, ydisp, lbl,
                    fontsize=max(label_fontsize - 1, 7),
                    ha='center', va='center',
                    color=base_txt_color2,
                    path_effects=pe_effects2,
                    zorder=6,
                )

        all_x.extend(ox)
        all_y.extend(oy)

    # Achsgrenzen aus allen sichtbaren Punkten ableiten
    if all_x and all_y:
        dx = max(all_x) - min(all_x)
        dy = max(all_y) - min(all_y)
        pad_x = dx * 0.06 if dx > 0 else 1.0
        pad_y = dy * 0.06 if dy > 0 else 1.0
        plt.xlim(min(all_x) - pad_x, max(all_x) + pad_x)
        plt.ylim(min(all_y) - pad_y, max(all_y) + pad_y)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'TSP / Orienteering Plot (expand={expand}, jitter={jitter})')
    plt.show()
