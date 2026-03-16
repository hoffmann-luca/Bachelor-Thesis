# exp_suite.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import Tuple, Dict, List
from math import isfinite
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm

# -------------------- Pfade ANPASSEN --------------------
RESULTS_PATH = Path(r"\features\v2\result_all.csv")            # rohe Runs (alle Seeds)
MODEL_RESULTS_PATH = Path(r"\features\v3\models\model_test_results.csv")  # Vorhersagen (Eval-Set)
FEATURES_PATH = r"\features\v2\train_data.csv"
OUTDIR = Path(r"\results\v3\analysis_out")
=======
RESULTS_PATH = Path(r"\features\v2\result_all.csv")            # rohe Runs (alle Seeds)
MODEL_RESULTS_PATH = Path(r"\features\v3_(aktuell)\models\model_test_results.csv")  # Vorhersagen (Eval-Set)
FEATURES_PATH = r"\features\v2\train_data.csv"
OUTDIR = Path(r"\results\v3_(aktuell)\analysis_out")
>>>>>>> Stashed changes:Trainingsdaten/Heuristiken/expPlots.py
# --------------------------------------------------------

#OUTDIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 200})

# -------------------- Hilfen --------------------

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stellt sicher, dass zentrale Spalten konsistente Typen haben
    (v.a. instance_id als str und limit_s als numerisch).
    """
    if "instance_id" in df:
        df["instance_id"] = df["instance_id"].astype(str)
    if "limit_s" in df:
        df["limit_s"] = pd.to_numeric(df["limit_s"], errors="coerce")
    return df


def _select_eval_runs(runs: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """
    Filtert das Runs-DataFrame so, dass nur (instance_id, limit_s)-Kombinationen
    übrig bleiben, die im Modell-Eval-Set (preds) vorkommen.
    """
    key = ["instance_id", "limit_s"]
    keys_eval = preds[key].drop_duplicates()
    merged = runs.merge(keys_eval, on=key, how="inner")
    return merged


def compute_vbs(eval_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Bestimmt den VBS (Virtual Best Solver) pro (Instanz, Limit):
    wählt den besten beobachteten Lauf über alle Algorithmen,
    mit Tiebreakern Remaining (↑) und Time (↓).
    """
    # Bester beobachteter Lauf pro (instanz, limit): Score↓ max, Remaining↑ min, Time↑ min
    runs_sorted = eval_runs.sort_values(
        ["instance_id", "limit_s", "score", "remaining", "used_time_s"],
        ascending=[True, True, False, True, True],
    )
    vbs = runs_sorted.groupby(["instance_id", "limit_s"]).head(1).copy()
    vbs = vbs.rename(columns={"algo": "vbs_algo", "score": "vbs_score"})
    return vbs[["instance_id", "limit_s", "vbs_algo", "vbs_score"]]


def compute_sbs(eval_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Bestimmt den SBS (Single Best Solver) pro Limit:
    berechnet den Algorithmus mit höchstem mittlerem Score im Eval-Set.
    """
    # SBS pro Limit: Algo mit höchstem mittleren Score im Eval-Set
    g = (
        eval_runs.groupby(["limit_s", "algo"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "mean_score"})
    )
    idx = g.groupby("limit_s")["mean_score"].idxmax()
    sbs = g.loc[idx].reset_index(drop=True)
    sbs = sbs.rename(columns={"algo": "sbs_algo", "mean_score": "sbs_score"})
    return sbs


def reduce_runs_to_means(eval_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Runs pro (instance_id, limit_s, algo):
    mittlerer Score, Remaining und Time über Seeds.
    """
    # mittlerer Score/Remaining/Time je (instanz,limit,algo)
    return (
        eval_runs.groupby(["instance_id", "limit_s", "algo"], as_index=False)
        .agg(score=("score", "mean"),
             remaining=("remaining", "mean"),
             used_time_s=("used_time_s", "mean"))
    )


def compute_best_algo_table(run_means: pd.DataFrame) -> pd.DataFrame:
    """
    Ermittelt den empirisch besten Algorithmus pro (Instanz, Limit),
    basierend auf den aggregierten Mittelwerten (Score als Hauptkriterium).
    """
    # best_algo nach Score (↓Remaining, ↓Time als Tiebreak)
    srt = run_means.sort_values(
        ["instance_id", "limit_s", "score", "remaining", "used_time_s"],
        ascending=[True, True, False, True, True]
    )
    best = srt.groupby(["instance_id", "limit_s"]).head(1).copy()
    best = best.rename(columns={"algo": "best_algo", "score": "best_score"})
    return best[["instance_id", "limit_s", "best_algo", "best_score"]]


def attach_model_score(run_means: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """
    Hängt zu jeder Modellvorhersage (pred_algo) den entsprechenden mittleren
    Score aus run_means an (Spalte model_score).
    """
    # Score für (instanz,limit,pred_algo) holen
    m = preds.merge(
        run_means.rename(columns={"algo": "pred_algo", "score": "model_score"}),
        on=["instance_id", "limit_s", "pred_algo"],
        how="left"
    )
    return m


def safe_ratio(a, b):
    """
    Berechnet a/b und gibt NaN zurück, falls b <= 0 oder nicht endlich ist.
    (Dient zur robusten Quotientenbildung in Auswertungen.)
    """
    a = float(a); b = float(b)
    if b <= 0 or not isfinite(b):
        return np.nan
    return a / b


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """
    Schätzt ein (1−alpha)-Konfidenzintervall für den Mittelwert von x
    mittels einfachem Bootstrap mit n_boot Resamples.
    """
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    means = []
    m = len(x)
    for _ in range(n_boot):
        sample = rng.choice(x, size=m, replace=True)
        means.append(sample.mean())
    lo = np.quantile(means, alpha/2)
    hi = np.quantile(means, 1 - alpha/2)
    return (float(lo), float(hi))


def mcnemar_test(y_true: np.ndarray, y_model: np.ndarray, y_sbs: np.ndarray) -> float:
    """
    Führt einen McNemar-Test auf Basis von 'korrekt/inkorrekt'-Vorhersagen durch:
    vergleicht Modell vs. SBS bezüglich Identifikation des best_algo.
    """
    # 2x2 Tabelle: model vs sbs KORREKT/FEHLERHAFT
    model_correct = (y_model == y_true).astype(int)
    sbs_correct = (y_sbs == y_true).astype(int)
    b = np.sum((model_correct == 1) & (sbs_correct == 0))  # nur Modell korrekt
    c = np.sum((model_correct == 0) & (sbs_correct == 1))  # nur SBS korrekt
    table = np.array([[0, c], [b, 0]])  # Diagonalen ignoriert
    try:
        res = mcnemar(table, exact=True)  # bei kleinen b,c exakte Variante
        return float(res.pvalue)
    except Exception:
        return np.nan


def add_wilcoxon_results(eval_df: pd.DataFrame, use_regret: bool = True) -> pd.DataFrame:
    """
    Führt Wilcoxon-Signed-Rank-Tests (1-seitig) pro (model_tag, limit_s) durch.

    eval_df-Spalten (mind.): instance_id, limit_s, model_tag, model_score, sbs_score, vbs_score
    use_regret=True: testet einseitig, ob Modell-REGRET < SBS-REGRET (besser)
    use_regret=False: testet einseitig, ob Modell-SCORE > SBS-SCORE (besser)

    Zusätzlich werden Effektstärken (r) und Holm-korrigierte p-Werte berechnet.
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
        d = sub["delta"].to_numpy()
        d = d[np.isfinite(d)]
        d_nonzero = d[np.abs(d) > 1e-12]
        n_eff = d_nonzero.size

        if n_eff < 10:
            p = np.nan
            z = np.nan
        else:
            stat = wilcoxon(
                d_nonzero,
                alternative=alternative,
                zero_method="wilcox",
                correction=True,
                method="auto",
            )
            p = stat.pvalue
            z = norm.ppf(1 - p) if np.isfinite(p) else np.nan  # einseitig
        r = z / np.sqrt(n_eff) if (n_eff > 0 and np.isfinite(z)) else np.nan

        rows.append({
            "model_tag": tag, "limit_s": L, "n_eff": n_eff,
            "wilcoxon_p": p, "wilcoxon_z": z, "effect_size_r": r,
            "direction": direction_label,
            "delta_mean": float(np.nanmean(d)) if d.size else np.nan,
            "delta_median": float(np.nanmedian(d)) if d.size else np.nan,
        })

    print(pd.DataFrame(rows).to_string(index=False))

    out = pd.DataFrame(rows).sort_values(["model_tag", "limit_s"]).reset_index(drop=True)

    # Multiple-Testing-Korrektur (Holm) je Modell über Limits
    adj = []
    for tag, sub in out.groupby("model_tag", sort=False):
        pvals = sub["wilcoxon_p"].to_numpy()
        mask = np.isfinite(pvals)
        p_adj = np.full_like(pvals, np.nan, dtype=float)
        if mask.any():
            _, p_holm, _, _ = multipletests(pvals[mask], method="holm")
            p_adj[mask] = p_holm
        sub = sub.copy()
        sub["wilcoxon_p_holm"] = p_adj
        def stars(p):
            if not np.isfinite(p): return ""
            return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."
        sub["sig_star"] = sub["wilcoxon_p_holm"].map(stars)
        adj.append(sub)

    return pd.concat(adj, ignore_index=True)

# -------------------- Hauptauswertung --------------------

def main():
    """
    Hauptpipeline für alle Auswertungen:
      - lädt Runs und Modellvorhersagen
      - bildet Eval-Subset und Aggregationen (VBS, SBS, run_means, best_algo)
      - joined_eval_table.csv als Basis-Tabelle
      - berechnet Primärmetriken, Spam-Check, Confusion-Matrizen
      - erzeugt Score-/VBS-Gap-/Within-δ-/Performance-Profil-Plots
      - führt Wilcoxon-Tests gegen SBS durch.
    """
    # Laden
    runs = _coerce_types(pd.read_csv(RESULTS_PATH))
    preds = _coerce_types(pd.read_csv(MODEL_RESULTS_PATH))

    # Eval-Subset der Runs
    eval_runs = _select_eval_runs(runs, preds)
    if eval_runs.empty:
        print("WARN: eval_runs leer – stimmen instance_id / limit_s in model_test_results.csv zu result_all.csv?")
    print(f"[INFO] Eval-Runs: {len(eval_runs):,} | Preds: {len(preds):,}")

    # Aggregationen
    vbs = compute_vbs(eval_runs)
    sbs = compute_sbs(eval_runs)
    run_means = reduce_runs_to_means(eval_runs)  # mittlere Algo-Scores je Instanz/Limit
    best = compute_best_algo_table(run_means)    # best_algo (für Accuracy/Confusion)

    # Modell-Score anhängen
    preds_scored = attach_model_score(run_means, preds)
    full = (preds_scored
            .merge(vbs, on=["instance_id", "limit_s"], how="left")
            .merge(sbs[["limit_s", "sbs_algo", "sbs_score"]], on="limit_s", how="left")
            .merge(best, on=["instance_id", "limit_s"], how="left"))

    # Grundchecks
    full.to_csv(OUTDIR / "joined_eval_table.csv", index=False)
    print(f"[OK] joined_eval_table.csv -> {OUTDIR}")

    # ---------- Primärmetriken je (model_tag, limit_s) ----------
    rows = []
    deltas_for_ci_by_group: Dict[Tuple[str, float], List[float]] = {}
    for (tag, L), grp in full.groupby(["model_tag", "limit_s"], dropna=False):
        n = len(grp)
        # Accuracy (gegen best_algo, das aus eval_runs abgeleitet wurde)
        acc_model = np.mean(grp["pred_algo"] == grp["best_algo"])

        # SBS-Accuracy (als Referenz)
        acc_sbs = np.mean(grp["sbs_algo"] == grp["best_algo"])
        acc_diff = acc_model - acc_sbs

        # Scorelagen (mittlere Punkte)
        ms = grp["model_score"].to_numpy(dtype=float)
        vs = grp["vbs_score"].to_numpy(dtype=float)
        ss = grp["sbs_score"].to_numpy(dtype=float)

        model_score_mean = np.nanmean(ms)
        vbs_score_mean = np.nanmean(vs)
        sbs_score_mean = np.nanmean(ss)

        # Regret (Abstand zum VBS)
        regret_abs_model = vs - ms
        regret_abs_sbs = vs - ss
        regret_abs_delta = np.nanmean(regret_abs_model) - np.nanmean(regret_abs_sbs)

        # within-δ% vom VBS
        rates = {}
        for delta_name, delta in [("within_0_1pct_vbs", 0.001),
                                  ("within_1pct_vbs", 0.01),
                                  ("within_5pct_vbs", 0.05)]:
            ok = []
            for a, b in zip(ms, vs):
                if not (isfinite(a) and isfinite(b)) or b <= 0:
                    ok.append(np.nan)
                else:
                    ok.append(a >= (1 - delta) * b)
            rates[delta_name] = np.nanmean(ok)

        # Score-Delta vs SBS (für CI & Wilcoxon)
        delta_score = ms - ss
        delta_mean = np.nanmean(delta_score)
        ci_lo, ci_hi = bootstrap_ci_mean(np.array(delta_score, dtype=float))

        deltas_for_ci_by_group[(str(tag), float(L) if pd.notna(L) else np.nan)] = list(np.array(delta_score, dtype=float))

        # McNemar (Genauigkeit Vergleich)
        mcn_p = mcnemar_test(
            grp["best_algo"].astype(str).to_numpy(),
            grp["pred_algo"].astype(str).to_numpy(),
            grp["sbs_algo"].astype(str).to_numpy(),
        )

        rows.append({
            "model_tag": tag, "limit_s": L, "n": n,
            "acc_model": acc_model, "acc_sbs": acc_sbs, "acc_diff": acc_diff,
            "model_score_mean": model_score_mean, "sbs_score_mean": sbs_score_mean, "vbs_score_mean": vbs_score_mean,
            "regret_abs_mean_model": np.nanmean(regret_abs_model),
            "regret_abs_mean_sbs": np.nanmean(regret_abs_sbs),
            "regret_abs_delta_model_minus_sbs": regret_abs_delta,
            "delta_score_mean": delta_mean, "delta_score_ci_lo": ci_lo, "delta_score_ci_hi": ci_hi,
            "within_0_1pct_vbs": rates["within_0_1pct_vbs"],
            "within_1pct_vbs": rates["within_1pct_vbs"],
            "within_5pct_vbs": rates["within_5pct_vbs"],
            "mcnemar_p": mcn_p
        })

    primary = pd.DataFrame(rows).sort_values(["limit_s", "model_tag"]).reset_index(drop=True)
    primary.to_csv(OUTDIR / "primary_metrics_summary.csv", index=False)
    print(f"[OK] primary_metrics_summary.csv -> {OUTDIR}")

    # ---------- Spam-Check & Confusion ----------
    spam_rows = []
    for (tag, L), grp in full.groupby(["model_tag", "limit_s"], dropna=False):
        pred_counts = Counter(grp["pred_algo"])
        total = sum(pred_counts.values())
        frac = {f"pred_frac_{k}": v / total for k, v in pred_counts.items()}
        spam_rows.append({"model_tag": tag, "limit_s": L, "n": total, **frac})
    spam = pd.DataFrame(spam_rows).sort_values(["limit_s", "model_tag"])
    spam.to_csv(OUTDIR / "spam_distribution.csv", index=False)

    # Confusion vs best_algo je Limit & Tag
    conf_dir = OUTDIR / "confusion_matrices"
    conf_dir.mkdir(exist_ok=True)
    for (tag, L), grp in full.groupby(["model_tag", "limit_s"], dropna=False):
        labels = sorted(set(grp["best_algo"].astype(str)) | set(grp["pred_algo"].astype(str)))
        idx = {lab: i for i, lab in enumerate(labels)}
        M = np.zeros((len(labels), len(labels)), dtype=int)
        for a, p in zip(grp["best_algo"].astype(str), grp["pred_algo"].astype(str)):
            M[idx[a], idx[p]] += 1
        dfM = pd.DataFrame(M, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
        dfM.to_csv(conf_dir / f"confusion_{tag}_limit_{L}.csv")

    # ---------- Plots (mit .to_numpy() gegen Pandas-Indexing-Bugs) ----------

    # 1) Scores vs Limit (VBS/SBS/Modelle)
    plt.figure()
    limits_sorted = sorted(primary["limit_s"].dropna().unique().tolist())
    # Referenzlinien aus dem ersten Tag ziehen (Werte sind pro Limit identisch)
    vbs_mean = [primary[primary["limit_s"] == L]["vbs_score_mean"].iloc[0] for L in limits_sorted]
    sbs_mean = [primary[primary["limit_s"] == L]["sbs_score_mean"].iloc[0] for L in limits_sorted]
    plt.plot(np.array(limits_sorted), np.array(vbs_mean), marker="o", label="VBS")
    plt.plot(np.array(limits_sorted), np.array(sbs_mean), marker="o", label="SBS")

    for tag, sub in primary.groupby("model_tag"):
        if str(tag) == "global":
            x = sub["limit_s"].to_numpy()
            y = sub["model_score_mean"].to_numpy()
            plt.plot(x, y, marker="o", label=str(tag))
        else:
            x = sub["limit_s"].to_numpy()
            y = sub["model_score_mean"].to_numpy()
            plt.plot(x, y, marker="o", linestyle="", label=str(tag))
    plt.xlabel("Zeitlimit [s]")
    plt.ylabel("Ø Score")
    plt.title("Score-Vergleich: VBS / SBS / Modelle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "primary_scores_vs_limit.png")
    plt.close()

    # 2) VBS-Gap (VBS − Score)
    plt.figure()
    for tag, sub in primary.groupby("model_tag"):
        if str(tag) == "global":
            x = sub["limit_s"].to_numpy()
            gap = (sub["vbs_score_mean"] - sub["model_score_mean"]).to_numpy()
            plt.plot(x, gap, marker="o", label=str(tag))
        else:
            x = sub["limit_s"].to_numpy()
            gap = (sub["vbs_score_mean"] - sub["model_score_mean"]).to_numpy()
            plt.plot(x, gap, marker="o", linestyle="", label=str(tag))
    sbs_gap = []
    for L in limits_sorted:
        row = primary[primary["limit_s"] == L].iloc[0]
        sbs_gap.append(float(row["vbs_score_mean"] - row["sbs_score_mean"]))
    plt.plot(np.array(limits_sorted), np.array(sbs_gap), linestyle="--", label="SBS-Gap")
    plt.xlabel("Zeitlimit [s]")
    plt.ylabel("VBS − ØScore")
    plt.title("Abstand zum VBS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "primary_vbs_gap.png")
    plt.close()

    # 3) Within-δ Raten
    for delta_col, title in [("within_0_1pct_vbs", "Within 0.1% VBS"),
                             ("within_1pct_vbs", "Within 1% VBS"),
                             ("within_5pct_vbs", "Within 5% VBS")]:
        plt.figure()
        for tag, sub in primary.groupby("model_tag"):
            if str(tag) == "global":
                x = sub["limit_s"].to_numpy()
                y = sub[delta_col].to_numpy()
                plt.plot(x, y, marker="o", label=str(tag))
            else:
                x = sub["limit_s"].to_numpy()
                y = sub[delta_col].to_numpy()
                plt.plot(x, y, marker="o", linestyle="", label=str(tag))
        plt.xlabel("Zeitlimit [s]")
        plt.ylabel("Anteil Instanzen")
        plt.title(title)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / f"primary_{delta_col}.png")
        plt.close()

    # 4) Performance-Profile (Score/VBS-Quotient ECDF) je Limit
    pp_dir = OUTDIR / "perf_profiles"
    pp_dir.mkdir(exist_ok=True)

    for L, grpL in full.groupby("limit_s"):
        # für jeden Tag: Liste der q = score/vbs_score
        tag_to_q = {}
        for tag, grp in grpL.groupby("model_tag"):
            q = []
            for a, b in zip(grp["model_score"].to_numpy(), grp["vbs_score"].to_numpy()):
                if isfinite(a) and isfinite(b) and b > 0:
                    q.append(max(0.0, min(1.0, a / b)))
            tag_to_q[str(tag)] = np.array(q, dtype=float)

        if not tag_to_q:
            continue

        plt.figure()
        xs = np.linspace(0.8, 1.0, 201)  # ab 0.8, damit Fokus auf gute Bereiche
        for tag, arr in tag_to_q.items():
            if len(arr) == 0:
                continue
            ys = np.array([(arr >= t).mean() for t in xs], dtype=float)
            plt.plot(xs, ys, label=tag)
        plt.xlabel("Score / VBS")
        plt.ylabel("Anteil Instanzen (≥ q)")
        plt.title(f"Performance-Profile (Limit={L:.1f}s)")
        plt.xlim(0.8, 1.0)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pp_dir / f"perf_profile_limit_{L:.1f}.png")
        plt.close()

    # 5) Signifikanz-Tabelle (McNemar & Wilcoxon) ist schon in primary_metrics_summary.csv enthalten

    wres = add_wilcoxon_results(
        full[["instance_id", "limit_s", "model_tag", "model_score", "sbs_score", "vbs_score"]],
        use_regret=True,  # (oder False, wenn du lieber Scores statt Regrets testest)
    )
    wres.to_csv(OUTDIR / "wilcoxon_by_model_limit.csv", index=False)
    print("Wilcoxon-Tabelle gespeichert:", OUTDIR / "wilcoxon_by_model_limit.csv")

    # --- Wilcoxon gegen SBS je Modell & Limit ---

    print("\n[FINISHED]")
    print(f"Outputs in: {OUTDIR}")
    print("- primary_metrics_summary.csv")
    print("- spam_distribution.csv")
    print("- confusion_matrices/*.csv")
    print("- primary_scores_vs_limit.png, primary_vbs_gap.png, primary_within_*.png")
    print("- perf_profiles/*.png")


if __name__ == "__main__":
    main()