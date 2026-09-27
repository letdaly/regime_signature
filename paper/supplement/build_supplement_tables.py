"""Write the data-driven tables of the supplementary material as LaTeX fragments.

Every number is read from the frozen outputs under ``results/``; nothing is
refitted.  Run from the repository root:

    .venv/bin/python paper/supplement/build_supplement_tables.py

The fragments land in ``paper/supplement/tables/`` and are \\input by
``paper/supplement/supplement.tex``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "tables"

DESIGNS = (("scale_only", "Design A"), ("matched_marginal", "Design B"))
LEARNERS = (("logistic", "Logit"), ("random_forest", "RF"), ("gbm", "GBM"))
AUGMENTED = (
    ("statistics_logsig", "statistics", r"Statistics $\to$ + log-sig."),
    ("statistics_rawsig", "statistics", r"Statistics $\to$ + raw sig."),
    ("b3_logsig", "b3", r"B3 $\to$ + log-sig."),
    ("b3_rawsig", "b3", r"B3 $\to$ + raw sig."),
)
CELL_LABELS = {
    "statistics": "Statistics",
    "statistics_logsig": "Statistics + log-sig.",
    "statistics_rawsig": "Statistics + raw sig.",
    "b3": "B3",
    "b3_logsig": "B3 + log-sig.",
    "b3_rawsig": "B3 + raw sig.",
}
CONTRAST_LABELS = {
    "statistics_logsig - statistics": r"Statistics $\to$ + log-sig.",
    "statistics_rawsig - statistics": r"Statistics $\to$ + raw sig.",
    "b3_logsig - b3": r"B3 $\to$ + log-sig.",
    "b3_rawsig - b3": r"B3 $\to$ + raw sig.",
}


def num(x, digits=2, signed=False):
    if x is None or not np.isfinite(x):
        return "--"
    text = "{:+.{d}f}".format(x, d=digits) if signed else "{:.{d}f}".format(x, d=digits)
    return "$" + text + "$" if text.startswith("-") else text


def interval(mean, low, high, digits=2):
    """Point estimate and bracketed interval, bold when the interval excludes zero."""

    body = "{} [{}, {}]".format(num(mean, digits, True), num(low, digits), num(high, digits))
    return r"\textbf{\boldmath " + body + "}" if (low > 0 or high < 0) else body


def pval(p):
    if p < 0.001:
        return "$<$0.001"
    return "{:.3f}".format(min(p, 1.0))


def t_interval(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    half = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
    return mean, mean - half, mean + half


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text.strip() + "\n")
    print("wrote", OUT / name)


def summary(path):
    frame = pd.read_csv(path)
    return frame.set_index("metric")


# ---------------------------------------------------------------------------
# Table S3 and S4: Holm-adjusted contrasts
# ---------------------------------------------------------------------------


def holm_tables():
    holm = pd.read_csv(RESULTS / "multiplicity" / "holm.csv")
    learner_names = dict(LEARNERS)
    learner_names["selected"] = "Selected"

    def rows(frame, extra_column=None):
        lines = []
        for _, row in frame.iterrows():
            cells = [row["unit"]]
            if extra_column is not None:
                cells.append(extra_column(row))
            cells += [
                CONTRAST_LABELS[row["contrast"]],
                interval(row["delta_points"], row["ci_low"], row["ci_high"]),
                pval(row["p_raw"]),
                pval(row["p_holm_family"]),
                pval(row["p_holm_global"]),
            ]
            lines.append(" & ".join(cells) + r" \\")
        return "\n".join(lines)

    classification = holm[holm["family"] == "classification"]
    market = holm[holm["family"] == "market"]
    body = (
        r"\multicolumn{7}{@{}l}{\textit{Panel A. Simulated classification (family of 24)}} \\" "\n"
        + rows(classification, lambda r: learner_names[r["learner"]])
        + "\n" r"\addlinespace" "\n"
        r"\multicolumn{7}{@{}l}{\textit{Panel B. Market forecasts, validation-selected learner (family of 8)}} \\" "\n"
        + rows(market, lambda r: "Selected")
    )
    write(
        "table_holm_classification_market.tex",
        r"""
\begin{tabular}{@{}lllcccc@{}}
\toprule
Design / market & Learner & Contrast & Gain [95\% CI] & $p$ & $p_{\mathrm{Holm}}^{\mathrm{family}}$ & $p_{\mathrm{Holm}}^{\mathrm{all}}$ \\
\midrule
"""
        + body
        + r"""
\bottomrule
\end{tabular}
""",
    )

    detection = holm[holm["family"] == "detection"]
    body = rows(detection, lambda r: "{:g}".format(r["budget"]))
    write(
        "table_holm_detection.tex",
        r"""
\begin{tabular}{@{}lllcccc@{}}
\toprule
Design & Target & Contrast & $\Delta$ DP [95\% CI] & $p$ & $p_{\mathrm{Holm}}^{\mathrm{family}}$ & $p_{\mathrm{Holm}}^{\mathrm{all}}$ \\
\midrule
"""
        + body
        + r"""
\bottomrule
\end{tabular}
""",
    )


# ---------------------------------------------------------------------------
# Table S5: logistic penalty audit
# ---------------------------------------------------------------------------


def penalty_table():
    lines = []
    for design, label in DESIGNS:
        narrow = pd.read_csv(RESULTS / "unified_grid" / design / "replications.csv").set_index("replication")
        wide = pd.read_csv(RESULTS / "unified_grid_logitc" / design / "replications.csv").set_index("replication")
        wide = wide.loc[narrow.index]
        lines.append(r"\multicolumn{4}{@{}l}{\textit{" + label + r"}} \\")
        for cell, cell_label in CELL_LABELS.items():
            column = "logistic_{}_test_ba".format(cell)
            a = 100 * narrow[column].to_numpy()
            b = 100 * wide[column].to_numpy()
            lines.append(
                " & ".join([cell_label, num(a.mean()), num(b.mean()), interval(*t_interval(b - a))]) + r" \\"
            )
        for sup, sub, pair in (
            ("b3", "statistics", r"B3 $-$ Statistics"),
            ("b3_logsig", "statistics_logsig", r"B3 + log-sig. $-$ Statistics + log-sig."),
            ("b3_rawsig", "statistics_rawsig", r"B3 + raw sig. $-$ Statistics + raw sig."),
        ):
            d3 = 100 * (narrow["logistic_{}_test_ba".format(sup)] - narrow["logistic_{}_test_ba".format(sub)])
            d9 = 100 * (wide["logistic_{}_test_ba".format(sup)] - wide["logistic_{}_test_ba".format(sub)])
            lines.append(
                " & ".join([r"\quad " + pair, interval(*t_interval(d3)), interval(*t_interval(d9)), ""]) + r" \\"
            )
        lines.append(r"\addlinespace")
    write(
        "table_logistic_penalty.tex",
        r"""
\begin{tabular}{@{}lccc@{}}
\toprule
Representation & Three-point grid & Nine-point grid & Nine minus three [95\% CI] \\
\midrule
"""
        + "\n".join(lines[:-1])
        + r"""
\bottomrule
\end{tabular}
""",
    )


# ---------------------------------------------------------------------------
# Table S6: order diagnostic
# ---------------------------------------------------------------------------


def order_table():
    specs = [
        ("{l}_statistics_logsig_minus_statistics_logsig_shuffled", r"Statistics + log-sig.: ordered $-$ shuffled"),
        ("{l}_statistics_rawsig_minus_statistics_rawsig_shuffled", r"Statistics + raw sig.: ordered $-$ shuffled"),
        ("{l}_b3_logsig_minus_b3_logsig_shuffled", r"B3 + log-sig.: ordered $-$ shuffled"),
        ("{l}_b3_rawsig_minus_b3_rawsig_shuffled", r"B3 + raw sig.: ordered $-$ shuffled"),
        ("{l}_statistics_logsig_shuffled_minus_statistics", r"Statistics $\to$ + shuffled log-sig."),
        ("{l}_statistics_rawsig_shuffled_minus_statistics", r"Statistics $\to$ + shuffled raw sig."),
        ("{l}_b3_logsig_shuffled_minus_b3", r"B3 $\to$ + shuffled log-sig."),
        ("{l}_b3_rawsig_shuffled_minus_b3", r"B3 $\to$ + shuffled raw sig."),
    ]
    lines = []
    for design, label in DESIGNS:
        s = summary(RESULTS / "unified_grid" / design / "summary.csv")
        lines.append(r"\multicolumn{4}{@{}l}{\textit{" + label + r"}} \\")
        for pattern, row_label in specs:
            cells = [row_label]
            for learner, _ in LEARNERS:
                r = s.loc["Delta_" + pattern.format(l=learner)]
                cells.append(interval(100 * r["mean"], 100 * r["ci_low"], 100 * r["ci_high"]))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    write(
        "table_order.tex",
        r"""
\begin{tabular}{@{}lccc@{}}
\toprule
Contrast & Logit & RF & GBM \\
\midrule
"""
        + "\n".join(lines[:-1])
        + r"""
\bottomrule
\end{tabular}
""",
    )


# ---------------------------------------------------------------------------
# Table S7: truncation order
# ---------------------------------------------------------------------------


def truncation_table():
    lines = []
    for design, label in DESIGNS:
        s = summary(RESULTS / "main_grid" / "truncation_{}".format(design) / "summary.csv")
        for learner, learner_label, prefix in (("random_forest", "RF", ""), ("logistic", "Logit", "logistic_")):
            cells = [label, learner_label]
            for metric in (
                "Delta_{p}combined_2_minus_statistics",
                "Delta_{p}combined_3_minus_statistics",
                "Delta_{p}combined_4_minus_statistics",
                "Delta_{p}combined_3_minus_combined_2",
                "Delta_{p}combined_4_minus_combined_3",
                "Delta_{p}noise_4_minus_statistics",
            ):
                r = s.loc[metric.format(p=prefix)]
                cells.append(interval(100 * r["mean"], 100 * r["ci_low"], 100 * r["ci_high"]))
            lines.append(" & ".join(cells) + r" \\")
    write(
        "table_truncation.tex",
        r"""
\begin{tabular}{@{}llcccccc@{}}
\toprule
 & & \multicolumn{3}{c}{Statistics + log-sig.\ minus Statistics} & \multicolumn{2}{c}{Step} & Noise \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(l){8-8}
Design & Learner & Level 2 (6) & Level 3 (14) & Level 4 (32) & $2\to3$ & $3\to4$ & control \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
""",
    )


# ---------------------------------------------------------------------------
# Tables S8 and S9: detection at matched false-alarm targets
# ---------------------------------------------------------------------------


def detection_tables():
    detectors = [("cusum", "none", "CUSUM"), ("hmm_filter", "none", "HMM filter")] + [
        (cell, "selected", label) for cell, label in CELL_LABELS.items()
    ]
    for (design, label), name in zip(DESIGNS, ("table_detection_a.tex", "table_detection_b.tex")):
        s = pd.read_csv(RESULTS / "unified_grid" / "detector_matched_fa" / design / "summary.csv")
        lines = []
        for target in (0.5, 1.0, 2.0, 5.0):
            lines.append(
                r"\multicolumn{5}{@{}l}{\textit{Target: " + "{:g}".format(target)
                + r" false alarms per 1,000 observations}} \\"
            )
            for detector, learner, detector_label in detectors:
                r = s[(s["detector"] == detector) & (s["learner"] == learner) & (s["target"] == target)].iloc[0]
                lines.append(
                    " & ".join(
                        [
                            detector_label,
                            "{:.2f}".format(r["false_alarms_per_1000"]),
                            "{:.1f} [{:.1f}, {:.1f}]".format(
                                100 * r["detection_probability"],
                                100 * r["detection_probability_ci_low"],
                                100 * r["detection_probability_ci_high"],
                            ),
                            "{:.1f}".format(100 * r["chance_detection_probability"]),
                            "{:.1f} [{:.1f}, {:.1f}]".format(
                                r["mean_delay"], r["mean_delay_ci_low"], r["mean_delay_ci_high"]
                            ),
                        ]
                    )
                    + r" \\"
                )
            lines.append(r"\addlinespace")
        write(
            name,
            r"""
\begin{tabular}{@{}lcccc@{}}
\toprule
Detector & Realized FA / 1,000 & DP (\%) [95\% CI] & Chance DP (\%) & Mean delay [95\% CI] \\
\midrule
"""
            + "\n".join(lines[:-1])
            + r"""
\bottomrule
\end{tabular}
""",
        )


# ---------------------------------------------------------------------------
# Table S10: balanced accuracy by time since the last change
# ---------------------------------------------------------------------------


def age_table():
    bins = (("0_24", "0--24"), ("25_49", "25--49"), ("50_99", "50--99"), ("100_plus", "100+"))
    columns = []
    for design, label in DESIGNS:
        s = summary(RESULTS / "main_grid" / "heston_{}".format(design) / "summary.csv")
        for prefix in ("", "logistic_"):
            columns.append(
                [
                    s.loc["Delta_BA_since_{}_{}combined_minus_statistics".format(b, prefix)]
                    for b, _ in bins
                ]
            )
    lines = []
    for i, (_, bin_label) in enumerate(bins):
        cells = [bin_label] + [
            interval(100 * col[i]["mean"], 100 * col[i]["ci_low"], 100 * col[i]["ci_high"]) for col in columns
        ]
        lines.append(" & ".join(cells) + r" \\")
    write(
        "table_time_since_change.tex",
        r"""
\begin{tabular}{@{}lcccc@{}}
\toprule
 & \multicolumn{2}{c}{Design A} & \multicolumn{2}{c}{Design B} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Observations since change & RF & Logit & RF & Logit \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
""",
    )


# ---------------------------------------------------------------------------
# Table S11: validation-selected learners in the market study
# ---------------------------------------------------------------------------


def market_learner_table():
    lines = []
    for market, label in (("us", "United States"), ("developed_ex_us", "Developed ex US")):
        folds = pd.read_csv(RESULTS / "market_french" / market / "folds.csv")
        chosen = folds[folds["selected"]]
        lines.append(r"\multicolumn{4}{@{}l}{\textit{" + label + r"}} \\")
        for cell in ("hmm",) + tuple(CELL_LABELS):
            picks = chosen[chosen["cell"] == cell]["learner"].value_counts()
            cell_label = "HMM probability block" if cell == "hmm" else CELL_LABELS[cell]
            lines.append(
                " & ".join([cell_label] + [str(int(picks.get(learner, 0))) for learner, _ in LEARNERS]) + r" \\"
            )
        lines.append(r"\addlinespace")
    write(
        "table_market_learners.tex",
        r"""
\begin{tabular}{@{}lccc@{}}
\toprule
Representation & Logit & RF & GBM \\
\midrule
"""
        + "\n".join(lines[:-1])
        + r"""
\bottomrule
\end{tabular}
""",
    )


if __name__ == "__main__":
    holm_tables()
    penalty_table()
    order_table()
    truncation_table()
    detection_tables()
    age_table()
    market_learner_table()
