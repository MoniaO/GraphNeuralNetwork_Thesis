"""Generate the three Task A thesis figures from frozen 14.08 artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "figures"
FINAL_JSON = ROOT / "outputs/taskA_FINAL_14.08.2026/FINAL_SUMMARY_14.08.2026.json"
CURVES = ROOT / "outputs/taskA_FINAL_14.08.2026/learning_curves/all_epoch_histories.csv"
STAGE_C_DIR = ROOT / "outputs/taskA_final_large_grid_11.08.2026/stage_c"
STAGE_A_CSV = ROOT / "outputs/taskA_final_large_grid_11.08.2026/stage_a/STAGE_A_BACKBONE_RANKING_11.08.2026.csv"

OFFICIAL = [
    "clean",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
]
ALL_SCEN = OFFICIAL + ["hidden_confounder"]
LABEL = {
    "clean": "Clean",
    "selection_bias": "Selection\nbias",
    "no_overlap": "No\noverlap",
    "noisy_documentation": "Noisy\ndocs",
    "multihospital": "Multi-\nhospital",
    "hidden_confounder": "Hidden\nconf.†",
}
COL_TRAIN = "#1e3a5f"      # navy — as in architecture / motif structural
COL_VALID = "#c2410c"      # orange — as in motif 40D / pair vectors
COL_TEST = "#0f766e"       # teal — as in motif encoders
COL_AUC_VALID = "#15803d"  # green
COL_AUC_TEST = "#86efac"   # light green
COL_MLP = "#1e3a5f"
COL_KAN = "#7A3E9D"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def load_final() -> pd.DataFrame:
    df = pd.DataFrame(json.loads(FINAL_JSON.read_text()))
    df = df[df["status"] == "ok"].copy()
    df = df.rename(columns={"training_seed": "seed"})
    return df


def load_curves() -> pd.DataFrame:
    df = pd.read_csv(CURVES)
    rows = []
    for run, g in df.groupby("run"):
        if not str(run).startswith("FINAL_"):
            continue
        rest = str(run)[len("FINAL_") :]
        encoder, scen_seed = rest.split("_", 1) if rest.startswith("mlp_") else (None, None)
        if rest.startswith("mlp_"):
            encoder = "mlp"
            scen_seed = rest[len("mlp_") :]
        elif rest.startswith("kan_shallow_"):
            encoder = "kan_shallow"
            scen_seed = rest[len("kan_shallow_") :]
        else:
            continue
        scenario, seed_s = scen_seed.rsplit("_seed", 1)
        g = g.copy()
        g["encoder"] = encoder
        g["scenario"] = scenario
        g["seed"] = int(seed_s)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def train_at_best(curves: pd.DataFrame) -> pd.DataFrame:
    out = []
    for keys, g in curves.groupby(["encoder", "scenario", "seed"]):
        best = int(g["best_epoch"].iloc[-1])
        row = g.loc[g["epoch"] == best]
        if row.empty:
            row = g.loc[[g["valid_auprc"].idxmax()]]
        out.append(
            {
                "encoder": keys[0],
                "scenario": keys[1],
                "seed": keys[2],
                "train_auprc_best": float(row["train_auprc"].iloc[0]),
            }
        )
    return pd.DataFrame(out)


def mean_sd(series: pd.Series) -> tuple[float, float]:
    return float(series.mean()), float(series.std(ddof=1)) if len(series) > 1 else 0.0


def _grouped_bars(ax, mlp, metrics, ylim, title, ylabel):
    x = np.arange(len(ALL_SCEN))
    width = 0.72 / len(metrics)
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2) * width
    for off, (col, lab, color) in zip(offsets, metrics):
        means, sds = [], []
        for scen in ALL_SCEN:
            m, s = mean_sd(mlp[mlp["scenario"] == scen][col])
            means.append(m)
            sds.append(s)
        bars = ax.bar(
            x + off,
            means,
            width=width * 0.92,
            yerr=sds,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            capsize=1.8,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
            label=lab,
            zorder=3,
        )
        bars[-1].set_hatch("///")
        bars[-1].set_edgecolor("#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[s] for s in ALL_SCEN])
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.yaxis.grid(True, lw=0.4, color="#E4E4E4", zorder=0)
    ax.set_axisbelow(True)


def figure_final_metrics(final: pd.DataFrame, curves: pd.DataFrame) -> None:
    mlp = final[final["encoder"] == "mlp"].copy()
    tr = train_at_best(curves)
    tr = tr[tr["encoder"] == "mlp"]
    mlp = mlp.merge(tr, on=["encoder", "scenario", "seed"], how="left")
    if mlp["train_auprc_best"].isna().any():
        raise RuntimeError("train AP at best epoch failed to merge")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.55), sharex=True)
    _grouped_bars(
        axes[0],
        mlp,
        [
            ("train_auprc_best", "Train (best epoch)", COL_TRAIN),
            ("valid_auprc", "Validation", COL_VALID),
            ("test_auprc", "Sealed test", COL_TEST),
        ],
        ylim=(0.84, 1.005),
        title="(a) AP",
        ylabel="AP",
    )
    _grouped_bars(
        axes[1],
        mlp,
        [
            ("valid_auc", "Validation", COL_AUC_VALID),
            ("test_auc", "Sealed test", COL_AUC_TEST),
        ],
        ylim=(0.92, 0.985),
        title="(b) AUC",
        ylabel="AUC",
    )
    fig.suptitle("Full model — metrics by scenario (mean ± SD, 5 seeds)", y=1.03, fontsize=11)
    fig.text(
        0.01,
        -0.04,
        r"$\dagger$ Hidden confounder is hatched and excluded from official macro averages. "
        "Train AP is taken at the validation-selected epoch.",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout()
    _save(fig, "fig_final_metrics_per_scenario")


def _epoch_mean_sd(curves: pd.DataFrame, encoder: str, col: str, scenarios: list[str]):
    sub = curves[(curves["encoder"] == encoder) & (curves["scenario"].isin(scenarios))]
    # First mean over seeds within scenario, then mean/sd over scenarios — more honest
    # for a 5-scenario protocol. Also return pooled seed-level band.
    recs = []
    for (scen, seed), g in sub.groupby(["scenario", "seed"]):
        recs.append(g[["epoch", col]].assign(scenario=scen, seed=seed))
    long = pd.concat(recs, ignore_index=True)
    # Align: for each scenario, mean over seeds at each epoch (nan if seed stopped)
    scen_means = []
    for scen, sg in long.groupby("scenario"):
        piv = sg.pivot_table(index="epoch", columns="seed", values=col, aggfunc="mean")
        scen_means.append(piv.mean(axis=1).rename(scen))
    mat = pd.concat(scen_means, axis=1).sort_index()
    mean = mat.mean(axis=1)
    sd = mat.std(axis=1, ddof=1).fillna(0.0)
    n = mat.notna().sum(axis=1)
    # Stop the line when fewer than 3 official scenarios remain
    keep = n >= 3
    return mean[keep], sd[keep]


def _plot_train_valid(ax, curves, encoder, title, col_train=COL_TRAIN, col_valid=COL_VALID):
    tr_m, tr_s = _epoch_mean_sd(curves, encoder, "train_auprc", OFFICIAL)
    va_m, va_s = _epoch_mean_sd(curves, encoder, "valid_auprc", OFFICIAL)
    ax.plot(tr_m.index, tr_m.values, color=col_train, lw=1.8, label="Train AP")
    ax.fill_between(tr_m.index, (tr_m - tr_s).values, (tr_m + tr_s).values, color=col_train, alpha=0.18, lw=0)
    ax.plot(va_m.index, va_m.values, color=col_valid, lw=1.8, label="Valid AP")
    ax.fill_between(va_m.index, (va_m - va_s).values, (va_m + va_s).values, color=col_valid, alpha=0.18, lw=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.set_xlim(1, 200)
    ax.set_ylim(0.45, 1.02)
    ax.set_title(title)
    ax.yaxis.grid(True, lw=0.4, color="#E4E4E4")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")


def figure_mlp_learning(curves: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    _plot_train_valid(ax, curves, "mlp", "Full model — train vs validation AP")
    fig.text(
        0.01,
        -0.02,
        "Mean ± 1 SD across five official scenarios (hidden confounder excluded). "
        "Each scenario is first averaged over five seeds.",
        fontsize=7.5,
        color="#444444",
    )
    _save(fig, "fig_final_mlp_learning")


def figure_mlp_vs_kan(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.45), sharey=True)
    _plot_train_valid(axes[0], curves, "mlp", "(a) MLP pair encoder")
    _plot_train_valid(
        axes[1],
        curves,
        "kan_shallow",
        "(b) KAN pair encoder",
        col_train="#5B4B8A",
        col_valid="#C43D6B",
    )
    axes[1].set_ylabel("")
    fig.suptitle("FINAL S10 — learning dynamics, MLP vs KAN", y=1.02, fontsize=11)
    fig.text(
        0.01,
        -0.04,
        "Same protocol: HGT + Fusion88 + S10. Mean ± 1 SD over five official scenarios "
        "(hidden confounder excluded); five seeds per scenario.",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout()
    _save(fig, "fig_final_mlp_vs_kan")


def print_milestones(final: pd.DataFrame) -> None:
    print("=== FINAL MLP per scenario (mean ± sd, 5 seeds) ===")
    mlp = final[final["encoder"] == "mlp"]
    for scen in ALL_SCEN:
        sub = mlp[mlp["scenario"] == scen]
        line = [scen]
        for col in ("valid_auprc", "test_auprc", "valid_auc", "test_auc"):
            m, s = mean_sd(sub[col])
            line.append(f"{m:.3f}±{s:.3f}")
        print(" | ".join(line))
    for enc in ("mlp", "kan_shallow"):
        sub = final[(final["encoder"] == enc) & (final["scenario"].isin(OFFICIAL))]
        per = sub.groupby("scenario")["valid_auprc"].mean()
        test = sub.groupby("scenario")["test_auprc"].mean()
        print(enc, "valid", float(per.mean()), "n_scen", len(per), per.to_dict())
        print(enc, "test ", float(test.mean()), test.to_dict())
        print(enc, "valid_auc", float(sub.groupby("scenario")["valid_auc"].mean().mean()))

    # Stage C S9/S10
    print("=== Stage C S9/S10 official macro ===")
    rows = []
    for scen in ALL_SCEN:
        path = STAGE_C_DIR / f"STAGE_C_SUMMARY_{scen}_hd4_11.08.2026.json"
        if scen == "clean":
            # clean full table has S0-S10; multi files are S9/S10 only except clean_hd4_FULL
            alt = STAGE_C_DIR / "STAGE_C_SUMMARY_clean_hd4_FULL_S0_S10_11.08.2026.json"
            data = json.loads(alt.read_text()) if alt.exists() else json.loads(path.read_text())
        else:
            data = json.loads(path.read_text())
        rows.extend(data)
    c = pd.DataFrame(rows)
    for var in ("S9_HCR_COMPACT36", "S10_HCR_FULL40"):
        sub = c[(c["variant"] == var) & (c["scenario"].isin(OFFICIAL))]
        per = sub.groupby("scenario")["valid_auprc"].mean()
        print(var, "official mean", float(per.mean()), per.round(4).to_dict())

    sa = pd.read_csv(STAGE_A_CSV)
    top = sa.sort_values("mean_valid_auprc", ascending=False).groupby("backbone").head(1)
    print("=== Stage A top-1 per backbone ===")
    print(top[["backbone", "mean_valid_auprc", "std_valid_auprc"]].to_string(index=False))


def main() -> None:
    _style()
    final = load_final()
    curves = load_curves()
    figure_final_metrics(final, curves)
    figure_mlp_learning(curves)
    figure_mlp_vs_kan(curves)
    print_milestones(final)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
