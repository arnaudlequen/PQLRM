"""
Plot OfficeWorld convergence traces.

Reads CSVs under <repo>/dataviz/data/<subfolder>/ and produces, per
subfolder, two PNGs in <repo>/dataviz/:

  - convergence_<subfolder>.png         (per-algo min-max normalized metric)
  - convergence_<subfolder>_raw.png     (raw metric, twin y-axis for QRM
                                         q-value sum vs PQL/PQLRM hypervolume)

Each CSV file in the subfolder is one labeled series. Series whose CSV has
a `hypervolume` column are treated as PQL/PQLRM; those with a `q_sum`
column are treated as QRM. Series are aggregated across seeds (mean +
95% CI) at matching step values.

Usage
-----
    python dataviz/plot_results.py                       # plot every subfolder
    python dataviz/plot_results.py --subfolder exp2      # one subfolder
    python dataviz/plot_results.py --max-steps 100000    # truncate x
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "dataviz" / "data"
DEFAULT_OUT_DIR = REPO_ROOT / "dataviz"


# ---------------------------------------------------------------------------
# TikZ / pgfplots helpers (ported from resources/CFXRL/dataviz/plot_results.py)
# ---------------------------------------------------------------------------

def _tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _tikz_id(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text))
    return cleaned if cleaned else "series"


def _hex_to_rgb_tuple(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return 0, 0, 0
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _stats_with_ci(data: pd.DataFrame, y_col: str) -> pd.DataFrame:
    means = data.groupby(["experiment", "x"], as_index=False)[y_col].mean()
    stds = (data.groupby(["experiment", "x"], as_index=False)[y_col]
                .std(ddof=1).rename(columns={y_col: "std"}))
    counts = (data.groupby(["experiment", "x"], as_index=False)[y_col]
                  .count().rename(columns={y_col: "count"}))
    stats = means.merge(stds, on=["experiment", "x"], how="left") \
                 .merge(counts, on=["experiment", "x"], how="left")
    stats["std"] = stats["std"].fillna(0.0)
    stats["ci95"] = 1.96 * stats["std"] / stats["count"].pow(0.5)
    stats["lower"] = stats[y_col] - stats["ci95"]
    stats["upper"] = stats[y_col] + stats["ci95"]
    return stats


def _write_pgfplots_line_tex(
    path: Path,
    data: pd.DataFrame,
    y_col: str,
    caption: str,
    y_label: str,
    color_map: dict[str, str],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    stats = _stats_with_ci(data, y_col)
    if y_min is not None:
        stats["lower"] = stats["lower"].clip(lower=y_min)
        stats["upper"] = stats["upper"].clip(lower=y_min)
    if y_max is not None:
        stats["lower"] = stats["lower"].clip(upper=y_max)
        stats["upper"] = stats["upper"].clip(upper=y_max)

    lines = [
        "% Requires in LaTeX preamble:",
        "% \\usepackage{pgfplots}",
        "% \\usepgfplotslibrary{fillbetween}",
        "% \\pgfplotsset{compat=1.18}",
    ]
    experiments = sorted(stats["experiment"].unique())
    color_name_map: dict[str, str] = {}
    for idx, exp in enumerate(experiments):
        cname = f"expcolor{idx}"
        color_name_map[exp] = cname
        r, g, b = _hex_to_rgb_tuple(color_map.get(exp, "#4c72b0"))
        lines.append(f"\\definecolor{{{cname}}}{{RGB}}{{{r},{g},{b}}}")

    lines.extend([
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "xlabel={Steps ($\\cdot 10^3$)},",
        f"ylabel={{{_tex_escape(y_label)}}},",
        "axis lines=left,",
        "xmin=0,",
        "grid=both,",
        "]",
    ])
    axis_end = lines.index("]")
    if y_min is not None:
        lines.insert(axis_end, f"ymin={float(y_min):.10g},")
        axis_end += 1
    else:
        lines.insert(axis_end, "ymin=0,")
        axis_end += 1
    if y_max is not None:
        lines.insert(axis_end, f"ymax={float(y_max):.10g},")

    for exp, group in stats.groupby("experiment"):
        group = group.sort_values("x")
        series_id = _tikz_id(exp)
        pgf_color = color_name_map.get(exp, "black")
        lower = " ".join(f"({float(x/1000.0):.10g},{float(y):.10g})"
                         for x, y in zip(group["x"], group["lower"]))
        upper = " ".join(f"({float(x/1000.0):.10g},{float(y):.10g})"
                         for x, y in zip(group["x"], group["upper"]))
        mean = " ".join(f"({float(x/1000.0):.10g},{float(y):.10g})"
                        for x, y in zip(group["x"], group[y_col]))
        lines.append(f"\\addplot[name path={series_id}low, draw=none] coordinates {{{lower}}};")
        lines.append(f"\\addplot[name path={series_id}up, draw=none] coordinates {{{upper}}};")
        lines.append(
            f"\\addplot[draw=none, fill={pgf_color}, fill opacity=0.15] "
            f"fill between[of={series_id}low and {series_id}up];"
        )
        lines.append(f"\\addplot[thick, draw={pgf_color}] coordinates {{{mean}}};")

    # Manual legend with explicit colored lines
    manual_legend = ["\\begin{tikzpicture}"]
    y = 0.0
    for exp in experiments:
        cname = color_name_map.get(exp, "black")
        manual_legend.append(
            f"\\draw[very thick, draw={cname}] (1.00,{y:.2f}) -- (2.20,{y:.2f});"
        )
        manual_legend.append(
            f"\\node[anchor=west] at (2.35,{y:.2f}) {{{_tex_escape(exp)}}};"
        )
        y -= 0.55
    manual_legend.append("\\end{tikzpicture}")

    lines.extend([
        "\\end{axis}",
        "\\end{tikzpicture}",
        "% Manual legend fallback with explicit colored lines:",
        *manual_legend,
        f"% caption: {_tex_escape(caption)}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_pgfplots_subplots_only_tex(
    path: Path,
    data: pd.DataFrame,
    y_col: str,
    y_label: str,
    color_map: dict[str, str],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    stats = _stats_with_ci(data, y_col)
    if y_min is not None:
        stats["lower"] = stats["lower"].clip(lower=y_min)
        stats["upper"] = stats["upper"].clip(lower=y_min)
    if y_max is not None:
        stats["lower"] = stats["lower"].clip(upper=y_max)
        stats["upper"] = stats["upper"].clip(upper=y_max)

    experiments = sorted(stats["experiment"].unique())
    lines = [
        "% Requires in LaTeX preamble:",
        "% \\usepackage{pgfplots}",
        "% \\usepgfplotslibrary{fillbetween}",
        "% \\pgfplotsset{compat=1.18}",
    ]
    color_name_map: dict[str, str] = {}
    for idx, exp in enumerate(experiments):
        cname = f"expcolor_sub{idx}"
        color_name_map[exp] = cname
        r, g, b = _hex_to_rgb_tuple(color_map.get(exp, "#4c72b0"))
        lines.append(f"\\definecolor{{{cname}}}{{RGB}}{{{r},{g},{b}}}")

    lines.extend([
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "width=0.95\\textwidth,",
        "height=0.38\\textwidth,",
        "xlabel={Steps ($\\cdot 10^3$)},",
        "axis y line*=right,",
        "yticklabel pos=right,",
        "ylabel={" + _tex_escape(y_label) + "},",
        "axis x line*=bottom,",
        "xmin=0,",
        "grid=both,",
        "]",
    ])
    axis_end = lines.index("]")
    if y_min is not None:
        lines.insert(axis_end, f"ymin={float(y_min):.10g},")
        axis_end += 1
    if y_max is not None:
        lines.insert(axis_end, f"ymax={float(y_max):.10g},")

    for idx, exp in enumerate(experiments):
        group = stats[stats["experiment"] == exp].sort_values("x")
        series_id = f"{_tikz_id(exp)}_{idx}"
        cname = color_name_map.get(exp, "black")
        lower = " ".join(f"({float(x/1000.0):.10g},{float(y):.10g})"
                         for x, y in zip(group["x"], group["lower"]))
        upper = " ".join(f"({float(x/1000.0):.10g},{float(y):.10g})"
                         for x, y in zip(group["x"], group["upper"]))
        mean = " ".join(f"({float(x/1000.0):.10g},{float(y):.10g})"
                        for x, y in zip(group["x"], group[y_col]))
        lines.append(f"\\addplot[name path={series_id}low, draw=none] coordinates {{{lower}}};")
        lines.append(f"\\addplot[name path={series_id}up, draw=none] coordinates {{{upper}}};")
        lines.append(
            f"\\addplot[draw=none, fill={cname}, fill opacity=0.15] "
            f"fill between[of={series_id}low and {series_id}up];"
        )
        lines.append(f"\\addplot[thick, draw={cname}] coordinates {{{mean}}};")

    lines.extend(["\\end{axis}", "\\end{tikzpicture}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, sep=";")
    except Exception as exc:
        print(f"  skip {path.name}: read failed ({exc})")
        return None
    if df.empty:
        return None

    if "hypervolume" in df.columns:
        metric_col = "hypervolume"
        kind = "hv"
    elif "q_sum" in df.columns:
        metric_col = "q_sum"
        kind = "qsum"
    else:
        print(f"  skip {path.name}: no hypervolume / q_sum column")
        return None

    x_col = "step" if "step" in df.columns else "episode"
    if x_col not in df.columns:
        print(f"  skip {path.name}: no step/episode column")
        return None

    df["x"] = pd.to_numeric(df[x_col], errors="coerce")
    df["y"] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    if df.empty:
        return None

    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(1).astype(int)
    else:
        df["seed"] = 1

    df["experiment"] = path.stem
    df["kind"] = kind

    if "computation_time" in df.columns:
        df["computation_time"] = pd.to_numeric(df["computation_time"], errors="coerce")

    cols = ["x", "y", "seed", "experiment", "kind"]
    if "computation_time" in df.columns:
        cols.append("computation_time")
    return df[cols]


def _normalize_per_series(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    mn = out.groupby("experiment")["y"].transform("min")
    mx = out.groupby("experiment")["y"].transform("max")
    denom = (mx - mn).replace(0, np.nan)
    out["y_norm"] = (out["y"] - mn) / denom
    out["y_norm"] = out["y_norm"].fillna(0.0)
    return out


def _forward_fill_to_grid(data: pd.DataFrame) -> pd.DataFrame:
    """Reindex every (experiment, seed) onto the union x grid and forward-fill.

    Algorithms that stopped logging earlier than others (e.g. a shorter
    training run, or a seed that crashed mid-way) get their last observed
    value held constant for every subsequent x in the grid. This makes the
    series comparable on the same x range without losing the late portion
    of the longer runs.

    Rows before a series' first observation remain unfilled and are dropped,
    so we never invent values *backward* in time -- only forward.
    """
    if data.empty:
        return data

    grid = np.sort(data["x"].unique())
    pieces: list[pd.DataFrame] = []
    has_ct = "computation_time" in data.columns

    for (experiment, kind, seed), group in data.groupby(["experiment", "kind", "seed"]):
        group = group.sort_values("x").drop_duplicates(subset="x")
        cols = ["y"] + (["computation_time"] if has_ct else [])
        reindexed = (
            group.set_index("x")[cols]
                 .reindex(grid)
                 .ffill()
                 .dropna(subset=["y"])
                 .reset_index()
        )
        reindexed["experiment"] = experiment
        reindexed["kind"] = kind
        reindexed["seed"] = seed
        pieces.append(reindexed)

    if not pieces:
        return data
    return pd.concat(pieces, ignore_index=True)


def _plot_subfolder(folder: Path, outdir: Path, max_steps: float | None) -> None:
    paths = sorted(folder.glob("*.csv"))
    if not paths:
        print(f"[{folder.name}] no CSVs")
        return

    frames = []
    for p in paths:
        df = _load_csv(p)
        if df is not None:
            frames.append(df)
    if not frames:
        print(f"[{folder.name}] no usable CSVs")
        return

    data = pd.concat(frames, ignore_index=True)
    data = (
        data.groupby(["experiment", "kind", "seed", "x"], as_index=False)
            .agg(y=("y", "mean"),
                 computation_time=("computation_time", "mean") if "computation_time" in data.columns else ("y", "size"))
    )
    if max_steps is not None:
        data = data[data["x"] <= max_steps].copy()
        if data.empty:
            print(f"[{folder.name}] nothing after max_steps={max_steps}")
            return

    data = _forward_fill_to_grid(data)
    data = _normalize_per_series(data)
    experiments = sorted(data["experiment"].unique())
    palette = sns.color_palette("deep", n_colors=max(len(experiments), 1)).as_hex()
    color_map = {e: c for e, c in zip(experiments, palette)}

    # --- normalized plot ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    data["xk"] = data["x"] / 1000.0
    sns.lineplot(
        data=data, x="xk", y="y_norm", hue="experiment",
        estimator="mean", errorbar=("ci", 95),
        linewidth=2.0, palette=color_map, legend=False, ax=ax,
    )
    ax.set_title(f"Convergence — {folder.name} (per-series min-max normalized)")
    ax.set_xlabel("Steps (x10^3)")
    ax.set_ylabel("Normalized metric")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.02, 1.05)
    handles = [Line2D([0], [0], color=color_map[e], lw=3) for e in experiments]
    ax.legend(handles, experiments, title="Series", fontsize=9, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    norm_path = outdir / f"convergence_{folder.name}.png"
    fig.savefig(norm_path, dpi=150)
    plt.close(fig)
    print(f"[{folder.name}] saved {norm_path}")

    # TikZ counterpart of the normalized plot
    tex_path = outdir / f"convergence_{folder.name}.tex"
    _write_pgfplots_line_tex(
        path=tex_path,
        data=data,
        y_col="y_norm",
        caption=f"Convergence - {folder.name} (per-series min-max normalized).",
        y_label="Normalized metric",
        color_map=color_map,
        y_min=0.0,
        y_max=1.05,
    )
    sub_tex_path = outdir / f"convergence_{folder.name}_subplots_only.tex"
    _write_pgfplots_subplots_only_tex(
        path=sub_tex_path,
        data=data,
        y_col="y_norm",
        y_label="Normalized metric",
        color_map=color_map,
        y_min=0.0,
        y_max=1.05,
    )
    print(f"[{folder.name}] saved {tex_path}")
    print(f"[{folder.name}] saved {sub_tex_path}")

    # --- raw plot with twin axis (HV left, q_sum right) ---
    fig, ax_hv = plt.subplots(figsize=(11, 5.5))
    ax_qsum = ax_hv.twinx()
    has_hv = (data["kind"] == "hv").any()
    has_qs = (data["kind"] == "qsum").any()

    for exp_name in experiments:
        sub = data[data["experiment"] == exp_name]
        ax = ax_hv if sub["kind"].iloc[0] == "hv" else ax_qsum
        sns.lineplot(
            data=sub, x="xk", y="y",
            estimator="mean", errorbar=("ci", 95),
            linewidth=2.0, color=color_map[exp_name], label=None, ax=ax,
        )

    ax_hv.set_xlabel("Steps (x10^3)")
    ax_hv.set_ylabel("Hypervolume (PQL/PQLRM)" if has_hv else "")
    ax_qsum.set_ylabel("Q-value sum at s0 (QRM)" if has_qs else "")
    ax_hv.set_xlim(left=0)
    ax_hv.set_title(f"Convergence — {folder.name} (raw metric)")
    handles = [Line2D([0], [0], color=color_map[e], lw=3) for e in experiments]
    ax_hv.legend(handles, experiments, title="Series", fontsize=9, ncol=2,
                 loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    raw_path = outdir / f"convergence_{folder.name}_raw.png"
    fig.savefig(raw_path, dpi=150)
    plt.close(fig)
    print(f"[{folder.name}] saved {raw_path}")

    # --- computation_time plot (optional) ---
    if "computation_time" in data.columns and data["computation_time"].notna().any():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ct = data.dropna(subset=["computation_time"])
        sns.lineplot(
            data=ct, x="xk", y="computation_time", hue="experiment",
            estimator="mean", errorbar=("ci", 95),
            linewidth=2.0, palette=color_map, legend=False, ax=ax,
        )
        ax.set_title(f"Computation Time — {folder.name}")
        ax.set_xlabel("Steps (x10^3)")
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(handles, experiments, title="Series", fontsize=9, ncol=2,
                  loc="upper center", bbox_to_anchor=(0.5, -0.18))
        fig.tight_layout()
        time_path = outdir / f"computation_time_{folder.name}.png"
        fig.savefig(time_path, dpi=150)
        plt.close(fig)
        print(f"[{folder.name}] saved {time_path}")

        time_tex_path = outdir / f"computation_time_{folder.name}.tex"
        _write_pgfplots_line_tex(
            path=time_tex_path,
            data=ct,
            y_col="computation_time",
            caption=f"Computation Time - {folder.name}, with 95\\% uncertainty bands.",
            y_label="Computation time (s)",
            color_map=color_map,
            y_min=0.0,
        )
        print(f"[{folder.name}] saved {time_tex_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Directory containing exp<N>/ subfolders of CSVs.")
    parser.add_argument("--subfolder", type=str, default=None,
                        help="Plot only this subfolder (e.g. exp2).")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="Where to write PNGs.")
    parser.add_argument("--style", type=str, default="whitegrid")
    parser.add_argument("--max-steps", type=float, default=None)
    args = parser.parse_args()

    sns.set_theme(style=args.style, context="talk")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.subfolder is not None:
        folders = [data_dir / args.subfolder]
        if not folders[0].is_dir():
            raise FileNotFoundError(folders[0])
    else:
        folders = sorted(p for p in data_dir.iterdir() if p.is_dir())
        if not folders:
            raise FileNotFoundError(f"No subfolders under {data_dir}")

    for folder in folders:
        _plot_subfolder(folder, outdir, args.max_steps)


if __name__ == "__main__":
    main()
