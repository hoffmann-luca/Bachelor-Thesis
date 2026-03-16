#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
secondary_plots.py

Erzeugt "sekundäre" Metriken + Plots aus einer per-Instanz Join-Tabelle.

Eingabe: joined_eval_table.csv (eine Zeile je Instanz & Limit)

Pflichtspalten:
    - instance_id
    - limit_s (float oder int)
    - model_tag (z.B. 'global', 'per-limit-20.0s', ...)
    - pred_algo        (vom Modell gewählte Heuristik)
    - model_score      (mittlerer Score der gewählten Methode auf dieser Instanz)
    - sbs_algo, sbs_score (Single Best Solver je Limit)
    - vbs_algo, vbs_score (Virtual Best je Instanz & Limit)

Optionale Spalten:
    - best_algo (falls vorhanden: "true" best_algo je Instanz & Limit;
                 z.B. aus prepare_results)
"""

import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, norm
from statsmodels.stats.multitest import multipletests


# -----------------------------
# IO
# -----------------------------

def load_joined(path: Path) -> pd.DataFrame:
    """
    Lädt die per-Instanz Join-Tabelle und sorgt dafür,
    dass limit_s als numerische Spalte vorliegt.
    """
    df = pd.read_csv(path)
    if "limit_s" in df.columns:
        df["limit_s"] = pd.to_numeric(df["limit_s"], errors="coerce")
    return df


# -----------------------------
# Aggregationen
# -----------------------------

def agg_win_tie_loss(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """
    Win/Tie/Loss (Score) gegen SBS je (model_tag, limit_s).

    - win_vs_sbs : Anteil Instanzen mit model_score >  sbs_score + eps
    - tie_vs_sbs : Anteil Instanzen mit |model_score - sbs_score| <= eps
    - loss_vs_sbs: Anteil Instanzen mit model_score <  sbs_score - eps
    """
    d = df.copy()
    d["win"]  = (d["model_score"] >  d["sbs_score"] + eps).astype(int)
    d["tie"]  = (np.abs(d["model_score"] - d["sbs_score"]) <= eps).astype(int)
    d["loss"] = (d["model_score"] <  d["sbs_score"] - eps).astype(int)
    out = (d.groupby(["model_tag", "limit_s"])
             .agg(n=("instance_id","count"),
                  win_vs_sbs=("win","mean"),
                  tie_vs_sbs=("tie","mean"),
                  loss_vs_sbs=("loss","mean"))
             .reset_index())
    return out


def _bootstrap_ci(x: np.ndarray,
                  func=np.mean,
                  n_boot: int = 2000,
                  alpha: float = 0.05,
                  seed: int = 0):
    """
    Einfacher Percentile-Bootstrap-CI für eine Kennzahl func(x).

    Rückgabe:
      (stat, lo, hi) mit:
        - stat:   func(x) auf Originaldaten
        - lo/hi:  alpha/2- und (1-alpha/2)-Quantil der Bootstrap-Verteilung
    """
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    stat = func(x)
    if x.size == 1:
        return stat, stat, stat
    boots = [func(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return stat, lo, hi


def agg_delta_score(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """
    ΔScore = model_score - sbs_score je (model_tag, limit_s),
    inkl. Bootstrap-95%-CI für den Mittelwert.
    """
    d = df.copy()
    d["delta_score"] = d["model_score"] - d["sbs_score"]
    rows = []
    for (tag, L), sub in d.groupby(["model_tag", "limit_s"], sort=True):
        arr = sub["delta_score"].to_numpy()
        mean, lo, hi = _bootstrap_ci(arr, np.mean, n_boot=n_boot, seed=seed)
        rows.append({"model_tag": tag, "limit_s": L,
                     "delta_score_mean": mean,
                     "delta_score_ci_lo": lo,
                     "delta_score_ci_hi": hi,
                     "n": int(sub.shape[0])})
    return pd.DataFrame(rows).sort_values(["model_tag","limit_s"]).reset_index(drop=True)


def agg_within_vbs(df: pd.DataFrame, deltas=(0.001, 0.01, 0.05)) -> pd.DataFrame:
    """
    Within-δ-VBS-Raten: Anteil Instanzen mit model_score >= (1-δ)*vbs_score.

    Gibt eine lange Tabelle zurück:
      Spalten: model_tag, limit_s, delta, within_vbs_rate, n
    """
    rows = []
    for (tag, L), sub in df.groupby(["model_tag", "limit_s"], sort=True):
        vs = sub["vbs_score"].to_numpy(dtype=float)
        ms = sub["model_score"].to_numpy(dtype=float)
        for delta in deltas:
            thresh = (1.0 - float(delta)) * vs
            ok = (ms >= thresh)
            rows.append({"model_tag": tag, "limit_s": L, "delta": float(delta),
                         "within_vbs_rate": float(np.mean(ok)),
                         "n": int(sub.shape[0])})
    return pd.DataFrame(rows).sort_values(["model_tag","limit_s","delta"]).reset_index(drop=True)


def agg_accuracy_vs_sbs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Klassifikationsgenauigkeit Modell vs. SBS je (model_tag, limit_s).

    Falls 'best_algo' vorhanden:
      - acc_model = P(pred_algo == best_algo)
      - acc_sbs   = P(sbs_algo  == best_algo)

    Sonst Fallback (Score-basiert):
      - acc_model = P(model_score >= sbs_score)
      - acc_sbs   = 1.0
    """
    d = df.copy()
    out_rows = []
    for (tag, L), sub in d.groupby(["model_tag","limit_s"], sort=True):
        if "best_algo" in sub.columns:
            acc_model = np.mean((sub["pred_algo"].astype(str).values == sub["best_algo"].astype(str).values))
            acc_sbs   = np.mean((sub["sbs_algo"].astype(str).values == sub["best_algo"].astype(str).values))
        else:
            acc_model = np.mean(sub["model_score"].values >= sub["sbs_score"].values)
            acc_sbs   = 1.0
        out_rows.append({
            "model_tag": tag, "limit_s": L, "n": int(sub.shape[0]),
            "acc_model": float(acc_model), "acc_sbs": float(acc_sbs),
            "acc_diff": float(acc_model - acc_sbs),
        })
    return pd.DataFrame(out_rows).sort_values(["model_tag","limit_s"]).reset_index(drop=True)


def add_wilcoxon_results(eval_df: pd.DataFrame, use_regret: bool = True) -> pd.DataFrame:
    """
    Wilcoxon-Signed-Rank-Tests pro (model_tag, limit_s).

    eval_df (mindestens Spalten):
      instance_id, limit_s, model_tag, model_score, sbs_score, vbs_score

    use_regret=True:
      - testet einseitig, ob Modell-REGRET < SBS-REGRET (besser)
        delta = (vbs - model) - (vbs - sbs)
    use_regret=False:
      - testet einseitig, ob Modell-SCORE > SBS-SCORE (besser)
        delta = model - sbs

    Rückgabe: Tabelle mit p-Werten, z-Scores, Effektstärken r und Holm-korrigierten p-Werten.
    """
    df = eval_df.copy()
    if use_regret:
        df["regret_model"] = df["vbs_score"] - df["model_score"]
        df["regret_sbs"]   = df["vbs_score"] - df["sbs_score"]
        df["delta"] = df["regret_model"] - df["regret_sbs"]  # < 0 ⇒ Modell besser
        alternative = "less"
        direction_label = "model < sbs (regret)"
    else:
        df["delta"] = df["model_score"] - df["sbs_score"]    # > 0 ⇒ Modell besser
        alternative = "greater"
        direction_label = "model > sbs (score)"

    rows = []
    for (tag, L), sub in df.groupby(["model_tag", "limit_s"], sort=True):
        d = sub["delta"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        d_nonzero = d[np.abs(d) > 1e-12]
        n_eff = d_nonzero.size

        if n_eff < 10:
            # zu wenig Paare -> kein Test
            p = np.nan
            z = np.nan
        else:
            stat = wilcoxon(
                d_nonzero,
                alternative=alternative,
                zero_method="wilcox",
                correction=True,
            )
            p = stat.pvalue
            z = norm.ppf(1 - p) if np.isfinite(p) else np.nan  # einseitige Richtung
        r = z / math.sqrt(n_eff) if (n_eff > 0 and np.isfinite(z)) else np.nan

        rows.append({
            "model_tag": tag, "limit_s": L, "n_eff": int(n_eff),
            "wilcoxon_p": p, "wilcoxon_z": z, "effect_size_r": r,
            "direction": direction_label,
            "delta_mean": float(np.nanmean(d)) if d.size else np.nan,
            "delta_median": float(np.nanmedian(d)) if d.size else np.nan,
        })

    out = pd.DataFrame(rows).sort_values(["model_tag", "limit_s"]).reset_index(drop=True)

    # Multiple-Testing-Korrektur (Holm) je Modell über Limits
    parts = []
    for tag, sub in out.groupby("model_tag", sort=False):
        pvals = sub["wilcoxon_p"].to_numpy(dtype=float)
        mask = np.isfinite(pvals)
        p_adj = np.full_like(pvals, np.nan, dtype=float)
        if mask.any():
            _, p_holm, _, _ = multipletests(pvals[mask], method="holm")
            p_adj[mask] = p_holm
        sub = sub.copy()
        sub["wilcoxon_p_holm"] = p_adj

        def stars(p):
            if not np.isfinite(p):
                return ""
            if p < 1e-3: return "***"
            if p < 1e-2: return "**"
            if p < 5e-2: return "*"
            return "n.s."

        sub["sig_star"] = [stars(p) for p in sub["wilcoxon_p_holm"].to_numpy()]
        parts.append(sub)

    return pd.concat(parts, ignore_index=True)


# -----------------------------
# Plots
# -----------------------------

def plot_win_tie_loss(win_df: pd.DataFrame, outdir: Path):
    """
    Gestapelte Balken für Win/Tie/Loss-Anteile pro (limit_s, model_tag).

    Ein Balken-Cluster pro Modell, darin farbige Segmente (win/tie/loss).
    """
    plt.figure(figsize=(7,4))
    limits = sorted(win_df["limit_s"].unique())
    tags = sorted(win_df["model_tag"].unique())
    width = 0.18
    x = np.arange(len(limits))

    for i, tag in enumerate(tags):
        sub = win_df[win_df["model_tag"] == tag].sort_values("limit_s")
        bottom = np.zeros_like(x, dtype=float)
        for comp, color in [("win_vs_sbs","#4daf4a"),
                            ("tie_vs_sbs","#999999"),
                            ("loss_vs_sbs","#e41a1c")]:
            vals = sub[comp].to_numpy(dtype=float)
            plt.bar(
                x + (i - len(tags)/2)*width + width/2,
                vals,
                width,
                bottom=bottom,
                label=f"{tag} · {comp.replace('_vs_sbs','')}" if comp=="win_vs_sbs" else None
            )
            bottom += vals

    plt.xticks(x, [str(L) for L in limits])
    plt.xlabel("Zeitlimit [s]")
    plt.ylabel("Anteil")
    plt.title("Win/Tie/Loss gegen SBS")
    plt.legend(ncol=max(1,len(tags)), fontsize=8)
    out = outdir / "sigplot_win_tie_loss.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Plot gespeichert:", out)


def plot_delta_score(delta_df: pd.DataFrame, wilcoxon_df: pd.DataFrame, outdir: Path):
    """
    Plot der Score-Differenz (Modell − SBS) mit 95%-CI und
    Signifikanz-Sternen aus dem Wilcoxon-Test.
    """
    plt.figure(figsize=(7,4))
    # vorbereiten: (model_tag, limit_s) -> "***" / "**" / "*" / "n.s."
    stars = { (r["model_tag"], r["limit_s"]): r["sig_star"] for _, r in wilcoxon_df.iterrows() }

    for tag, sub in delta_df.groupby("model_tag"):
        sub = sub.sort_values("limit_s")
        y = sub["delta_score_mean"].to_numpy()
        x = sub["limit_s"].to_numpy()
        ylo = sub["delta_score_ci_lo"].to_numpy()
        yhi = sub["delta_score_ci_hi"].to_numpy()
        yerr = np.vstack([y - ylo, yhi - y])
        plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, label=tag)
        for xi, yi, L in zip(x, y, sub["limit_s"]):
            star = stars.get((tag, L), "")
            if star and star != "n.s.":
                plt.text(xi, yi, star, ha="center", va="bottom", fontsize=9)

    plt.axhline(0.0, color="k", linewidth=1)
    plt.xlabel("Zeitlimit [s]")
    plt.ylabel("ΔScore (Modell − SBS)")
    plt.title("Score-Differenz vs. SBS (95%-CI, Sterne = Wilcoxon-Holm)")
    plt.legend()
    out = outdir / "sigplot_delta_score_ci.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Plot gespeichert:", out)


def plot_within_vbs(within_df: pd.DataFrame, outdir: Path):
    """
    Linienplot der Within-δ-VBS-Raten je Modell und Limit.
    """
    plt.figure(figsize=(7,4))
    for tag, sub in within_df.groupby("model_tag"):
        for delta, ss in sub.groupby("delta"):
            ss = ss.sort_values("limit_s")
            plt.plot(
                ss["limit_s"].to_numpy(),
                ss["within_vbs_rate"].to_numpy(),
                marker="o",
                linestyle="-",
                label=f"{tag} · ≤{int(delta*1000)/10 if delta<0.01 else int(delta*100)}% VBS-Lücke"
            )
    plt.ylim(0,1.01)
    plt.xlabel("Zeitlimit [s]")
    plt.ylabel("Anteil Instanzen")
    plt.title("Within-δ-VBS-Raten")
    plt.legend(ncol=2, fontsize=8)
    out = outdir / "sigplot_within_vbs_rates.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Plot gespeichert:", out)


def plot_accuracy_vs_sbs(acc_df: pd.DataFrame, outdir: Path):
    """
    Plot der Modell-Accuracy vs. SBS-Accuracy (nur zur Einordnung,
    da eigentliche Optimierung scoring-basiert erfolgt).
    """
    plt.figure(figsize=(7,4))
    for tag, sub in acc_df.groupby("model_tag"):
        sub = sub.sort_values("limit_s")
        plt.plot(sub["limit_s"].to_numpy(), sub["acc_model"].to_numpy(), marker="o", label=f"{tag} (Modell)")
    # SBS-Linie (falls pro Limit nicht konstant 1.0)
    if not np.allclose(acc_df["acc_sbs"].dropna().unique(), 1.0):
        for tag, sub in acc_df.groupby("model_tag"):
            sub = sub.sort_values("limit_s")
            plt.plot(sub["limit_s"].to_numpy(), sub["acc_sbs"].to_numpy(), linestyle="--", label=f"{tag} (SBS)")
    else:
        limits = sorted(acc_df["limit_s"].unique())
        plt.plot(limits, [1.0]*len(limits), linestyle="--", label="SBS")
    plt.ylim(0,1.01)
    plt.xlabel("Zeitlimit [s]")
    plt.ylabel("Accuracy")
    plt.title("Accuracy (nur Einordnung)")
    plt.legend()
    out = outdir / "sigplot_accuracy_vs_sbs.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Plot gespeichert:", out)


def _order_models(df: pd.DataFrame) -> list[str]:
    """
    Hilfsfunktion: bevorzugte Sortierung der model_tag-Namen,
    Rest hinten anhängen.
    """
    pref = ["global","per-limit-1.0s","per-limit-5.0s","per-limit-10.0s","per-limit-20.0s","per-limit-40.0s"]
    have = [m for m in pref if m in df["model_tag"].unique()]
    rest = [m for m in df["model_tag"].unique() if m not in have]
    return have + rest


def _nice_model(tag: str) -> str:
    """
    Erzeugt lesbareren Namen aus einem model_tag (nur kosmetisch).
    """
    if tag == "global":
        return "global"
    if tag.startswith("per-limit-"):
        val = tag.split("-")[-1].replace("s","")
        return f"per-limit {val}s"
    return tag


def compute_within_vbs_rates(full: pd.DataFrame,
                             deltas=(0.001, 0.01, 0.05)) -> pd.DataFrame:
    """
    Liefert "tidy" Tabelle mit Within-VBS-Raten:

      Spalten: [model_tag, limit_s, delta_key, rate]
      delta_key ist '≤0.1%', '≤1%' oder '≤5%'.
    """
    out = []
    for d in deltas:
        key = f"≤{(100*d):g}%"
        ok = full["model_score"] >= (1.0 - d) * full["vbs_score"]
        tmp = (full.assign(within=ok.astype(int))
                     .groupby(["model_tag", "limit_s"])["within"]
                     .mean()
                     .reset_index()
                     .rename(columns={"within": "rate"}))
        tmp["delta_key"] = key
        out.append(tmp)
    rates = pd.concat(out, ignore_index=True)
    return rates


def plot_within_vbs_heatmaps(rates: pd.DataFrame, outdir: Path):
    """
    Heatmaps je δ:
      - Zeilen: Modelle
      - Spalten: limit_s
      - Wert: Within-VBS-Rate (als Prozent annotiert)
    """
    model_order = sorted(rates["model_tag"].unique())
    limit_order = sorted(rates["limit_s"].unique())

    for key, sub in rates.groupby("delta_key"):
        pivot = (sub.pivot(index="model_tag", columns="limit_s", values="rate")
                    .reindex(index=model_order, columns=limit_order))
        fig, ax = plt.subplots(figsize=(1.2*len(limit_order)+3, 0.6*len(model_order)+2))
        im = ax.imshow(pivot.values, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(limit_order)))
        ax.set_xticklabels(limit_order)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(model_order)
        ax.set_xlabel("Zeitlimit [s]")
        ax.set_title(f"Within-VBS-Rate (δ {key})")
        # Zellen annotieren
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{100*val:.0f}%", ha="center", va="center", fontsize=9, color="black")
        fig.colorbar(im, ax=ax, label="Anteil Instanzen")
        fig.tight_layout()
        fig.savefig(outdir / f"heatmap_within_vbs_{key.replace('≤','le_').replace('%','pct')}.png", dpi=200)
        plt.close(fig)


# ---------- Win/Tie/Loss (robust & vollständig) ----------

def compute_wtl(full: pd.DataFrame, tie_eps=0.0) -> pd.DataFrame:
    """
    Win/Tie/Loss gegen SBS je (model_tag, limit_s) in "tidy"-Form.

    tie_eps: absolute Toleranz auf Score-Differenz für "tie".
    Rückgabe:
      Spalten: [model_tag, limit_s, outcome, share]
      outcome ∈ {'win','tie','loss'}
    """
    diff = full["model_score"] - full["sbs_score"]
    outcome = np.where(
        diff > tie_eps, "win",
        np.where(diff < -tie_eps, "loss", "tie")
    )
    tmp = (full.assign(outcome=outcome)
                .groupby(["model_tag", "limit_s", "outcome"])
                .size()
                .rename("n")
                .reset_index())
    # auf Anteile normieren
    total = tmp.groupby(["model_tag", "limit_s"])["n"].transform("sum")
    tmp["share"] = tmp["n"] / total
    # fehlende Kategorien auffüllen (damit für jeden (model_tag,limit_s) alle outcomes da sind)
    idx = pd.MultiIndex.from_product(
        [sorted(full["model_tag"].unique()),
         sorted(full["limit_s"].unique()),
         ["win","tie","loss"]],
        names=["model_tag","limit_s","outcome"]
    )
    tmp = (tmp.set_index(["model_tag","limit_s","outcome"])
              .reindex(idx, fill_value=0)
              .reset_index())
    return tmp


def plot_wtl_facet(wtl: pd.DataFrame, outdir: Path):
    """
    Facettierter Win/Tie/Loss-Plot:

    - ein Subplot pro Zeitlimit
    - je Subplot: gestapelte Balken für alle Modelle (win/tie/loss-Anteile)
    """
    limits = sorted(wtl["limit_s"].unique())
    models = sorted(wtl["model_tag"].unique())
    colors = {"win":"tab:green", "tie":"tab:orange", "loss":"tab:red"}

    ncol = min(3, len(limits))
    nrow = int(np.ceil(len(limits)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5*ncol, 3.6*nrow), squeeze=False, sharey=True)
    for ax, L in zip(axes.ravel(), limits):
        sub = (wtl[wtl["limit_s"]==L]
               .pivot(index="model_tag", columns="outcome", values="share")
               .reindex(models)
               .fillna(0.0))
        x = np.arange(len(models))
        bottom = np.zeros(len(models))
        for outcome in ["win","tie","loss"]:
            vals = sub[outcome].values if outcome in sub.columns else np.zeros(len(models))
            ax.bar(x, vals, bottom=bottom, color=colors[outcome],
                   label=outcome if L==limits[0] else None)
            bottom += vals
        ax.set_title(f"{L:.0f}s")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.2)
    # gemeinsame Legende oben
    handles = [plt.Rectangle((0,0),1,1,color=colors[k]) for k in ["win","tie","loss"]]
    labels = ["win", "tie", "loss"]
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.supylabel("Anteil")
    fig.supxlabel("Modelle")
    fig.tight_layout(rect=(0,0,1,0.92))
    fig.savefig(outdir / "wtl_facet.png", dpi=200)
    plt.close(fig)


# -----------------------------
# MAIN
# -----------------------------

def main():
    """
    CLI-Einstieg:

    - lädt joined_eval_table.csv
    - berechnet sekundäre Metriken (Win/Tie/Loss, ΔScore, Within-VBS, Accuracy, Wilcoxon)
    - schreibt resultierende CSVs
    - erzeugt entsprechende Plots im Ausgabeverzeichnis
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", type=str, default="joined_eval_table.csv",
                    help="Pfad zur per-Instanz Join-Tabelle (CSV)")
    ap.add_argument("--outdir", type=str, default="analysis_out_secondary",
                    help="Output-Verzeichnis für CSVs und Plots")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="Anzahl Bootstrap-Samples für CI von ΔScore")
    ap.add_argument("--seed", type=int, default=0, help="Zufallsseed für Bootstrap")
    ap.add_argument("--deltas", type=str, default="0.001,0.01,0.05",
                    help="Kommagetrennte δ-Werte für Within-VBS, z.B. '0.001,0.01,0.05'")
    args = ap.parse_args()

    joined_path = Path(args.joined)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_joined(joined_path)

    # 1) Win/Tie/Loss (einfache Aggregationstabelle)
    wtl = agg_win_tie_loss(df)
    wtl.to_csv(outdir / "win_tie_loss_by_limit.csv", index=False)

    # 2) ΔScore + CI
    dsum = agg_delta_score(df, n_boot=args.bootstrap, seed=args.seed)
    dsum.to_csv(outdir / "delta_score_summary.csv", index=False)

    # 3) Within-δ-VBS-Raten
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    wvbs = agg_within_vbs(df, deltas=tuple(deltas))
    wvbs.to_csv(outdir / "within_vbs_rates.csv", index=False)

    # 4) Accuracy vs SBS (nur Einordnung)
    acc = agg_accuracy_vs_sbs(df)
    acc.to_csv(outdir / "accuracy_vs_sbs.csv", index=False)

    # 5) Wilcoxon (einseitig) auf Regret (Modell < SBS ⇒ besser)
    wres = add_wilcoxon_results(
        df[["instance_id","limit_s","model_tag","model_score","sbs_score","vbs_score"]],
        use_regret=True
    )
    wres.to_csv(outdir / "wilcoxon_by_model_limit.csv", index=False)

    # ---------------- PLOTS ----------------
    plot_win_tie_loss(wtl, outdir)
    plot_delta_score(dsum, wres, outdir)
    plot_within_vbs(wvbs, outdir)
    plot_accuracy_vs_sbs(acc, outdir)

    # 1) Within-VBS als Heatmaps je δ (mit delta_key)
    rates = compute_within_vbs_rates(df, deltas=(0.001, 0.01, 0.05))
    plot_within_vbs_heatmaps(rates, outdir)

    # 2) Win / Tie / Loss vollständig (tidy + Facettenplot)
    wtl_full = compute_wtl(df, tie_eps=0.0)
    plot_wtl_facet(wtl_full, outdir)

    # Kurze textuelle Zusammenfassung in die Konsole
    print("\n=== Sekundäre Metriken — Kurzüberblick ===")
    print("ΔScore (Mean) je Modell/Limit:\n",
          dsum.pivot(index="limit_s", columns="model_tag", values="delta_score_mean"))
    print("\nWithin-1%-VBS-Rate je Modell/Limit:\n",
          wvbs[wvbs["delta"]==0.01].pivot(index="limit_s", columns="model_tag", values="within_vbs_rate"))
    print("\nWilcoxon-Holm p-Werte je Modell/Limit:\n",
          wres.pivot(index="limit_s", columns="model_tag", values="wilcoxon_p_holm"))


if __name__ == "__main__":
    main()
