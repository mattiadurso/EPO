"""Plotting / aggregation utilities for the benchmark notebooks.

Reads pose-AUC and NVS results from on-disk JSON dumps produced by
``benchmark_pose`` and ``test.py`` runs and renders the comparison plots
used in the paper. Imported almost exclusively by ``benchmark.ipynb``.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output

from helpers.benchmark_pose import eval_colmap_model_all_scenes


def read_results(
    dataset, target_folder, models, thr=5, round_to=1, remove=None, full=False
):
    """Aggregate per-scene pose-AUC results for a list of models on a dataset.

    Args:
        dataset: Dataset name (matches a subfolder under ``base_target``).
        target_folder: Sub-path under each model's run directory.
        models: List of model run names to compare.
        thr: AUC threshold (degrees) to report.
        round_to: Number of decimals to round results to.
        remove: Scene names to exclude from aggregation. ``None`` is treated
            as the empty list.
        full: If True, read from the ``benchmarks_full`` tree.

    Returns:
        ``pd.DataFrame`` with one column per model and one row per scene.
    """
    if remove is None:
        remove = []
    base_target = f"/home/mattia/Desktop/datasets/{dataset}"
    if dataset == "imc":
        base_target = (
            "/home/mattia/Desktop/Repos/posebench/benchmarks_2D/imc/data/phototourism/"
        )
    base_repo = "/home/mattia/Desktop/Repos/batchsfm/benchmarks"
    base_repo = (
        "/home/mattia/Desktop/Repos/batchsfm/benchmarks_full" if full else base_repo
    )
    scenes = os.listdir(base_target)

    # Read results
    dfs = {}
    for name in models:
        dfs[name] = eval_colmap_model_all_scenes(
            target_path=base_target,
            target_folder=target_folder,
            input_path=f"{base_repo}/{name}/{dataset}",
            input_folder="sparse",
            return_df=True,
            thrs=[thr],
            round_to=8,
            verbose=False,
        )

    keys = sorted(list(dfs.keys()))
    for key in keys:
        dfs[key].columns = [
            f"{col}_{key}" if "auc" in col else col for col in dfs[key].columns
        ]

    # Read timings
    total_timings = {}
    for model in models:
        if model not in total_timings:
            total_timings[f"{model}"] = {}
        for scene in scenes:
            benchmarks = "benchmarks_full/" if full else "benchmarks/"
            recon_path = f"{benchmarks}{model}/{dataset}/{scene}"
            if not os.path.exists(recon_path):
                continue
            if scene not in total_timings[f"{model}"]:
                total_timings[f"{model}"][scene] = None
                try:
                    with open(f"{recon_path}/sparse/timings.txt") as f:
                        lines = f.readlines()
                    total_timings[f"{model}"][scene] = float(lines[-1].split()[1])
                except FileNotFoundError:
                    total_timings[f"{model}"][scene] = None

    df = pd.concat([dfs[key][f"auc@{thr}_{key}"] for key in keys], axis=1)

    df = pd.concat([df, pd.DataFrame(total_timings)], axis=1)
    for col in df.columns:
        if col.startswith("vggt_edge"):
            df[col] += df["vggt"]  # include base model time

    df.columns = [
        col.replace("_", "+") if "auc" not in col else col for col in df.columns
    ]

    for rem in remove:
        df.drop(rem, inplace=True, errors="ignore")

    df.loc["mean"] = df.mean(numeric_only=True)
    df = df.round(round_to)

    # clean up cell prints
    clear_output()
    print("Results:")
    display(df)

    return df


def plot_auc5_with_time(df, dataset, models, thr, ignore_time=False):
    """Render the AUC@thr-vs-runtime comparison figure used in the paper.

    Args:
        df: DataFrame produced by :func:`read_results`.
        dataset: Dataset name (used as a plot title; recognised aliases:
            ``"tt"`` → "Tanks & Temples", ``"scannetpp"`` → "Scannet++ v2").
        models: Ordered list of model names to plot.
        thr: AUC threshold whose columns to read.
        ignore_time: If True, plot AUC only (no time axis).
    """
    dataset = "Tanks & Temples" if dataset == "tt" else dataset
    dataset = "Scannet++ v2" if dataset == "scannetpp" else dataset

    # --- 1. Data Preparation ---
    df_auc = df[[col for col in df.columns if f"auc@{thr}" in col]].copy()

    # Clean column names
    clean_cols = [col.replace(f"auc@{thr}_", "") for col in df_auc.columns]
    df_auc.columns = clean_cols

    # Combine into one DataFrame
    df_combined = df_auc.copy()

    if not ignore_time:
        # Read the Time row directly from the 'mean' row of the dataframe
        time_dict = {}
        if "mean" in df.index:
            for col in clean_cols:
                # Map column name back to time column name (usually + instead of _)
                time_col = col.replace("_", "+")
                # Get the value from the mean row
                time_dict[col] = (
                    df.loc["mean", time_col] if time_col in df.columns else 0.0
                )
        else:
            # Fallback if 'mean' row is missing (though prompt implies it exists)
            time_dict = {col: 0.0 for col in clean_cols}

        time_values = [time_dict.get(col, 0) for col in clean_cols]
        # Create single row DF
        df_time_row = pd.DataFrame(
            [time_values], columns=clean_cols, index=["Tot Time"]
        )

        # Combine AUC and Time
        df_combined = pd.concat([df_auc, df_time_row])

        # --- 1b. Define vggt_time for ratio calculation ---
        # Get the 'vggt' time
        vggt_time = time_dict.get("vggt", 1.0)
        if vggt_time == 0:
            vggt_time = 1.0  # Ensure it is not zero if it was found but is zero

    # --- 2. Masking for Dual Axis ---
    df_p1 = df_combined.copy()

    if not ignore_time:
        df_p1.loc["Tot Time"] = np.nan  # Hide Time on Left Axis

        df_p2 = df_combined.copy()
        df_p2.iloc[:-1] = np.nan  # Hide AUC on Right Axis

    # --- 3. Plotting ---
    fig, ax1 = plt.subplots(
        figsize=(max(len(df_combined.index) * len(df_combined.columns) / 2, 4), 4)
    )

    # Plot AUC (Left Axis)
    df_p1.plot(kind="bar", ax=ax1, width=0.8)

    # Adjust Left Y-Axis to fit text (Add 15% margin)
    max_auc = df_p1.max().max()
    if not np.isnan(max_auc):
        ax1.set_ylim(0, max_auc * 1.15)

    # --- 4. Styling & Separator ---
    ax1.set_ylabel(f"AUC@{thr}")

    ax1.set_title(f"AUC@{thr} & Time - {dataset[:1].upper() + dataset[1:]} Dataset")

    if not ignore_time:
        # Black Dashed Line between AUC rows and Time row
        sep_x = len(df_auc) - 0.5
        ax1.axvline(x=sep_x, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    # --- ROTATION CHANGE HERE ---
    # Rotate 45 degrees and align right so the text ends at the tick mark
    ax1.set_xticklabels(df_combined.index, rotation=45, ha="right")

    # --- 5. Add Ratio Text on Top of Time Bars (The requested change) ---

    # Plot Time (Right Axis)
    if not ignore_time:
        ax2 = ax1.twinx()
        df_p2.plot(kind="bar", ax=ax2, width=0.8, legend=False)
        ax2.set_ylabel("Total Time (s)")

        # Adjust Right Y-Axis to fit text (Add 15% margin)
        max_time = df_p2.max().max()
        if not np.isnan(max_time):
            ax2.set_ylim(0, max_time * 1.15)

        # The bars on ax2 are from df_p2, which only has values in the 'Tot Time' row.
        # We only care about bars after the separator line (i.e., the 'Tot Time' group).
        for rect in ax2.patches:
            current_time = rect.get_height()
            # Check if the bar is part of the 'Tot Time' group (x-position after separator)
            # and has a non-zero height (we only annotate visible bars)
            if rect.get_x() > sep_x and current_time > 0:
                # Calculate Ratio
                ratio = current_time / vggt_time
                ratio_text = f"{ratio:.2f}x"

                # Add text annotation
                # rect.get_x() + rect.get_width() / 2 is the center of the bar
                ax2.text(
                    rect.get_x() + rect.get_width() / 2,
                    current_time,
                    ratio_text,
                    ha="center",
                    va="bottom",  # Align text just above the bar
                    fontsize=8,
                )

    # --- 6. Add AUC Score on Top of Mean Bars ---
    # Get the index position of "mean" row
    mean_idx = (
        list(df_combined.index).index("mean") if "mean" in df_combined.index else None
    )

    if mean_idx is not None:
        # ax1 patches are grouped by column, so we need to find bars at mean_idx position
        num_rows = len(df_combined.index)
        num_cols = len(df_combined.columns)

        for i, rect in enumerate(ax1.patches):
            # Calculate which row this bar belongs to
            row_idx = i % num_rows
            col_idx = i // num_rows

            if row_idx == mean_idx:
                auc_value = rect.get_height()
                if not np.isnan(auc_value) and auc_value > 0:
                    ax1.text(
                        rect.get_x() + rect.get_width() / 2,
                        auc_value,
                        f"{auc_value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        # fontweight="bold",
                    )

    # --- 7. Legend Placement ---
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout()
    # Save the figure instead of showing it
    # plt.savefig("auc5_with_time_plot.png")
    plt.show()


def read_results_nvs(
    nvs_results_path="/home/mattia/HDD_Fast/gaussian-splatting-results/data",
    exclude_metrics=(),
    column_order=None,
):
    """Args:
    round_to (int): Decimal places for rounding.
    exclude_metrics (tuple): Metrics to exclude from results.
    column_order (list): List of method names to set column order.
                       If None, uses default sorting (GT first, then alphabetical).
    """
    # nvs_results_path = "/home/mattia/Desktop/Repos/gaussian-splatting/data"
    data = []

    if not os.path.exists(nvs_results_path):
        return pd.DataFrame()

    methods = os.listdir(nvs_results_path)

    for method in methods:
        method_path = os.path.join(nvs_results_path, method)
        if not os.path.isdir(method_path):
            continue

        for scene in os.listdir(method_path):
            res_json = os.path.join(method_path, scene, "3dgs/results.json")
            if os.path.exists(res_json):
                with open(res_json) as f:
                    res = json.load(f)

                # Capitalize scene name here
                formatted_scene = scene.replace("_", " ").title()

                metrics = res.get("ours_30000", {})
                entry = {"method": method, "scene": formatted_scene, **metrics}
                data.append(entry)

    df = pd.DataFrame(data)
    if exclude_metrics:
        df.drop(columns=list(exclude_metrics), errors="ignore", inplace=True)

    # Pivot and swap levels to (Method, Metric)
    metric_cols = [c for c in df.columns if c not in ["method", "scene"]]
    df = df.pivot(index="scene", columns="method", values=metric_cols)
    df.columns = df.columns.swaplevel(0, 1)

    # Get unique methods and metrics
    unique_methods = df.columns.get_level_values(0).unique().tolist()
    unique_metrics = df.columns.get_level_values(1).unique().tolist()

    # Determine method order
    if column_order is None:
        # Default: GT first, then alphabetical
        if "GT" in unique_methods:
            ordered_methods = ["GT"] + sorted([m for m in unique_methods if m != "GT"])
        else:
            ordered_methods = sorted(unique_methods)
    else:
        # Use provided order, append any missing methods at the end
        ordered_methods = [m for m in column_order if m in unique_methods]
        remaining = [m for m in unique_methods if m not in column_order]
        ordered_methods.extend(sorted(remaining))

    # Custom metric order: PSNR, SSIM, LPIPS
    metric_order = ["PSNR", "SSIM", "LPIPS"]
    ordered_metrics = [m for m in metric_order if m in unique_metrics]
    # Append any remaining metrics not in the predefined order
    ordered_metrics.extend([m for m in unique_metrics if m not in metric_order])

    # Build new column order: for each method, include all metrics
    new_columns = []
    for method in ordered_methods:
        for metric in ordered_metrics:
            if (method, metric) in df.columns:
                new_columns.append((method, metric))

    # Reindex with the new column order
    df = df[new_columns]

    df.sort_index(axis=0, inplace=True)

    # Append Mean row
    df_final = pd.concat([df, pd.DataFrame(df.mean(), columns=["Mean"]).T])
    df_final.index.name = "Scene"

    return df_final


def latexfy(
    df, caption="Comparison of Results", label="tab:results", time_round=1, auc_round=1
):
    """Generates a LaTeX table from the dataframe output of read_results.

    Args:
        df (pd.DataFrame): Output from read_results.
        caption (str): Table caption.
        label (str): Table label.
        time_round (int): Number of decimal places for Time columns (default 1).
        auc_round (int): Number of decimal places for AUC columns (default 3).

    Returns:
        str: A string containing the formatted LaTeX table.
    """
    # 1. Identify Model Pairs (AUC col + Time col)
    auc_cols = [c for c in df.columns if "auc" in c]

    try:
        thr = auc_cols[0].split("_")[0].split("@")[1]
    except IndexError:
        thr = "?"

    model_pairs = []

    for auc_col in auc_cols:
        prefix = f"auc@{thr}_"
        if not auc_col.startswith(prefix):
            continue

        raw_model_suffix = auc_col[len(prefix) :]
        time_col = raw_model_suffix.replace("_", "+")

        if time_col in df.columns:
            display_name = time_col.upper()
            model_pairs.append({"name": display_name, "auc": auc_col, "time": time_col})

    # 2. Setup LaTeX Header
    n_models = len(model_pairs)
    col_setup = "l " + "cc " * n_models

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"    \centering")
    # latex.append(r"    % \setlength{\tabcolsep}{4pt}")
    latex.append(r"    \resizebox{\linewidth}{!}{")
    latex.append(f"        \\begin{{tabular}}{{{col_setup}}}")
    latex.append(r"            \toprule")

    # Row 1: Model Names
    header_names = ["            "]
    for m in model_pairs:
        header_names.append(f"& \\multicolumn{{2}}{{c}}{{\\textbf{{{m['name']}}}}}")
    latex.append(" ".join(header_names) + " \\\\")

    # Row 2: CMidrules
    cmids = []
    for i in range(n_models):
        start = 2 + (i * 2)
        end = start + 1
        cmids.append(f"\\cmidrule(lr){{{start}-{end}}}")
    latex.append("            " + " ".join(cmids))

    # Row 3: Metrics
    metrics_row = [r"            \textbf{Scene}"]
    for _ in model_pairs:
        metrics_row.append(f"& AUC@{thr} $\\uparrow$ & Time $\\downarrow$")
    latex.append(" ".join(metrics_row) + " \\\\")
    latex.append(r"            \midrule")

    # 3. Data Rows
    indices = [i for i in df.index if str(i).lower() != "mean"]

    for idx in indices:
        # Format scene name: remove underscores and capitalize
        formatted_idx = str(idx).replace("_", " ").title()
        row_str = [f"            {formatted_idx}"]

        # --- Logic Change: Calculate 'Best' only among NON-VGGT models ---
        comparison_pairs = [m for m in model_pairs if m["name"] != "VGGT"]

        row_auc_values = [df.loc[idx, m["auc"]] for m in comparison_pairs]
        row_time_values = [df.loc[idx, m["time"]] for m in comparison_pairs]

        valid_auc = [v for v in row_auc_values if pd.notnull(v)]
        valid_time = [v for v in row_time_values if pd.notnull(v)]

        best_auc = max(valid_auc) if valid_auc else -1
        best_time = min(valid_time) if valid_time else float("inf")
        # -----------------------------------------------------------------

        for m in model_pairs:
            val_auc = df.loc[idx, m["auc"]]
            val_time = df.loc[idx, m["time"]]

            # Helper to format and optionally bold
            def format_cell(val, best_val, precision, is_vggt, bold_enabled=True):
                """Render a numeric cell, bolding the best non-VGGT value."""
                if pd.isnull(val):
                    return "-"
                txt = f"{val:.{precision}f}"
                if bold_enabled and not is_vggt and val == best_val:
                    txt = f"\\textbf{{{txt}}}"
                return txt

            is_vggt = m["name"] == "VGGT"

            str_auc = format_cell(
                val_auc, best_auc, auc_round, is_vggt, bold_enabled=False
            )
            str_time = format_cell(
                val_time, best_time, time_round, is_vggt, bold_enabled=False
            )

            row_str.append(f"& {str_auc} & {str_time}")

        latex.append(" ".join(row_str) + " \\\\")

    # 4. Mean Row
    if "mean" in df.index:
        latex.append(r"            \midrule")
        row_str = [r"            \textbf{Mean}"]

        for m in model_pairs:
            val_auc = df.loc["mean", m["auc"]]
            val_time = df.loc["mean", m["time"]]

            # Logic: Bold all means EXCEPT VGGT (based on your previous example style)
            is_vggt = m["name"] == "VGGT"

            if pd.notnull(val_auc):
                txt_auc = f"{val_auc:.{auc_round}f}"
                if not is_vggt:
                    txt_auc = f"\\textbf{{{txt_auc}}}"
            else:
                txt_auc = "-"

            if pd.notnull(val_time):
                txt_time = f"{val_time:.{time_round}f}"
            else:
                txt_time = "-"

            row_str.append(f"& {txt_auc} & {txt_time}")

        latex.append(" ".join(row_str) + " \\\\")

    # 5. Footer
    latex.append(r"            \bottomrule")
    latex.append(r"        \end{tabular}")
    latex.append(r"    }")
    latex.append(f"    \\caption{{{caption}}}")
    latex.append(f"    \\label{{{label}}}")
    latex.append(r"\end{table}")

    return "\n".join(latex)
