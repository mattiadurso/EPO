import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

import sys

sys.path.append("/home/mattia/Desktop/Repos/posebench/benchmarks_3D")
from benchmark_pose import eval_colmap_model_all_scenes, eval_colmap_model


def read_results(dataset, target_folder, models, thr=5, round_to=1, remove=[]):
    base_target = f"/home/mattia/Desktop/datasets/{dataset}"
    base_repo = "/home/mattia/Desktop/Repos/batchsfm/benchmarks"
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
            round_to=round_to,
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
            recon_path = f"benchmarks/{model}/{dataset}/{scene}"
            if not os.path.exists(recon_path):
                continue
            if scene not in total_timings[f"{model}"]:
                total_timings[f"{model}"][scene] = None
                try:
                    with open(f"{recon_path}/sparse/timings.txt", "r") as f:
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

    df.loc["mean"] = df.mean()
    df = df.round(round_to)

    # clean up cell prints
    clear_output()
    print("Results:")
    display(df)

    return df


def plot_auc5_with_time(df, dataset, models, thr, ignore_time=False):
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

    ax1.set_title(f"AUC@{thr} & Time - {dataset[:1].upper()+dataset[1:]} Dataset")

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
