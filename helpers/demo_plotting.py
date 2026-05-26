"""Plotting helpers for the demo notebook.

Renders the convergence / stability summary plot used in ``demo.ipynb`` from
either a live ``EPO`` instance or a ``training_logs.json`` dump produced by
``EPO.to_colmap`` so the notebook cell collapses to a single function call.
"""

import json
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection


def _to_floats(values):
    """Cast a possibly tensor-containing iterable to a 1-D float ``np.ndarray``."""
    return np.array(
        [v.item() if torch.is_tensor(v) else v for v in values], dtype=float
    )


def _smoothed(values, window):
    """Trailing rolling-mean of ``values`` with the given window size."""
    return np.convolve(values, np.ones(window) / window, mode="valid")


def check_convergence_pure(values, mode, window, tol):
    """Free-function clone of ``EPO.check_convergence``.

    Mirrors the in-class semantics so this module works against a JSON dump
    where no live ``EPO`` instance is available.

    Args:
        values: Sequence of per-step scalars (pose change or loss).
        mode: ``"loss"`` switches to relative-change checking; anything else
            treats ``values`` as already-absolute changes.
        window: Smoothing window size in steps.
        tol: Tolerance applied to every smoothed value in the window.

    Returns:
        ``True`` once all ``window`` smoothed values lie below ``tol``.
    """
    required = 2 * window - 1
    if len(values) < required:
        return False
    recent = _to_floats(values[-required:])
    smoothed = _smoothed(recent, window)
    if mode == "loss":
        smoothed = np.abs(np.diff(smoothed) / (np.abs(smoothed[:-1]) + 1e-8))
    return bool(np.all(smoothed < tol))


def _extract(source):
    """Normalise ``source`` into the dict of arrays the plotter consumes.

    ``source`` may be:
        * an ``EPO`` instance (uses its in-memory lists),
        * a path to a ``training_logs.json`` dump,
        * a dict already shaped like that JSON.
    """
    if isinstance(source, str):
        with open(source) as f:
            source = json.load(f)

    if isinstance(source, Mapping):
        changes = source["list_changes"]
        steps = np.asarray(changes.get("steps") or np.arange(source["steps_actual"]))
        return {
            "steps": steps,
            "max_changes": _to_floats(changes["max"]),
            "q_changes": _to_floats(changes.get("q", [])),
            "t_changes": _to_floats(changes.get("t", [])),
            "raw_loss": _to_floats(source["list_loss"]),
            "auc": {int(k): list(v) for k, v in source["list_auc"]["auc"].items()},
            "auc_steps": list(source["list_auc"].get("steps", [])),
            "auc_saving_freq": source.get("auc_saving_freq", 50),
            "lr_list": source.get("list_lr", {}),
        }

    # Treat anything else as an EPO instance.
    return {
        "steps": np.asarray(source.changes["steps"]),
        "max_changes": _to_floats(source.changes["max"]),
        "q_changes": _to_floats(source.changes.get("q", [])),
        "t_changes": _to_floats(source.changes.get("t", [])),
        "raw_loss": _to_floats(source.loss_list),
        "auc": {int(k): list(v) for k, v in source.auc_list["auc"].items()},
        "auc_steps": list(source.auc_list.get("steps", [])),
        "auc_saving_freq": source.auc_saving_freq,
        "lr_list": getattr(source, "lr_list", {}),
    }


def _rising_edges(flags):
    """Indices where ``flags`` flips from ``False`` to ``True``."""
    f = np.asarray(flags, dtype=bool)
    return list(np.where(f[1:] & ~f[:-1])[0] + 1)


def _draw_convergence_marks(ax, steps, values, mode, window, tol, color, label, ymax):
    """Draw a vertical line at the first step where the convergence flag rises."""
    flags = [
        check_convergence_pure(values[: i + 1], mode, window, tol)
        for i in range(len(values))
    ]
    edges = _rising_edges(flags)
    if not edges:
        return
    ax.vlines(
        steps[edges[0]],
        0,
        ymax,
        color=color,
        linestyle="--",
        linewidth=2,
        zorder=4,
        label=label,
    )


def plot_training_convergence(
    source,
    *,
    window_pose=25,
    window_pose2=50,
    window_loss=25,
    tol_pose=0.5,
    tol_pose2=0.1,
    tol_loss=5e-4,
    auc_threshold=5,
    figsize=(14, 4),
    title="Training Convergence & Stability Metrics",
):
    """Plot pose-change, scaled loss, AUC and convergence markers vs. step.

    Args:
        source: An ``EPO`` instance, a path to ``training_logs.json``, or a
            dict shaped like that JSON.
        window_pose: Smoothing window for the first pose-change tolerance band.
        window_pose2: Smoothing window for the second (tighter) band.
        window_loss: Smoothing window for the loss-based convergence check.
        tol_pose: Pose-change tolerance for ``window_pose``; drawn as a
            horizontal line and used to mark the first satisfying step.
        tol_pose2: Pose-change tolerance for ``window_pose2``.
        tol_loss: Relative-change tolerance for the loss convergence check.
        auc_threshold: AUC bucket (degrees) plotted on the secondary axis.
        figsize: ``matplotlib`` figure size.
        title: Axes title.

    Returns:
        ``(fig, ax)`` so callers can save or further customise.
    """
    data = _extract(source)
    steps = data["steps"]
    max_changes = data["max_changes"]
    raw_loss = data["raw_loss"]
    auc = data["auc"]

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Max Change (Delta)", color="blue")
    ax1.plot(
        steps, max_changes, alpha=0.15, color="blue", zorder=1, label="Raw Max Change"
    )

    if len(max_changes) >= window_pose:
        s = _smoothed(max_changes, window_pose)
        ax1.plot(
            steps[window_pose - 1 :],
            s,
            color="darkblue",
            linewidth=2,
            zorder=3,
            label=f"Smoothed (W={window_pose})",
        )
    if len(max_changes) >= window_pose2:
        s = _smoothed(max_changes, window_pose2)
        ax1.plot(
            steps[window_pose2 - 1 :],
            s,
            color="teal",
            linewidth=1.5,
            linestyle="--",
            zorder=3,
            label=f"Smoothed (W={window_pose2})",
        )

    ax1.axhline(
        tol_pose, color="red", linestyle=":", alpha=0.6, label=f"Tol {tol_pose}"
    )
    ax1.axhline(
        tol_pose2, color="green", linestyle=":", alpha=0.6, label=f"Tol {tol_pose2}"
    )

    ymax = max_changes.max() if max_changes.size else 1.0
    _draw_convergence_marks(
        ax1,
        steps,
        max_changes,
        "pose",
        window_pose,
        tol_pose,
        "red",
        f"Reached Tol {tol_pose}",
        ymax,
    )
    _draw_convergence_marks(
        ax1,
        steps,
        max_changes,
        "pose",
        window_pose2,
        tol_pose2,
        "green",
        f"Reached Tol {tol_pose2}",
        ymax,
    )

    if raw_loss.size:
        loss_x = steps[: len(raw_loss)]
        l_min, l_max = raw_loss.min(), raw_loss.max()
        m_min, m_max = max_changes.min(), max_changes.max()
        denom = (l_max - l_min) or 1.0
        scaled_loss = (raw_loss - l_min) / denom * (m_max - m_min) + m_min
        ax1.plot(
            loss_x,
            scaled_loss,
            color="purple",
            alpha=0.5,
            linestyle=":",
            label="Scaled Loss Trend",
        )
        _draw_convergence_marks(
            ax1,
            steps,
            raw_loss,
            "loss",
            window_loss,
            tol_loss,
            "black",
            f"Loss converged (<{tol_loss})",
            ymax,
        )

    auc_curve = auc.get(auc_threshold, [])
    if auc_curve:
        ax2 = ax1.twinx()
        ax2.set_ylabel(f"AUC@{auc_threshold}", color="orange")
        auc_steps = np.arange(len(auc_curve)) * data["auc_saving_freq"]
        ax2.plot(
            auc_steps,
            auc_curve,
            color="orange",
            marker="o",
            markersize=4,
            alpha=0.8,
            label=f"AUC@{auc_threshold}",
        )
        for i, (x, y) in enumerate(zip(auc_steps, auc_curve, strict=False)):
            offset = 0.5 if i % 2 == 0 else -1.5
            ax2.text(
                x,
                y + offset,
                f"{y:.2f}",
                color="black",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        y_min, y_max = min(auc_curve), max(auc_curve)
        ax2.set_ylim(y_min - 0.02, y_max * 1.02)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(
            h1 + h2,
            l1 + l2,
            loc="upper left",
            fontsize="small",
            ncol=2,
            bbox_to_anchor=(1.05, 1),
        )
    else:
        ax1.legend(loc="upper left", fontsize="small", ncol=2, bbox_to_anchor=(1.05, 1))

    ax1.set_title(title)
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax1


def plot_training_summary(
    source,
    *,
    auc_thresholds=(1, 3, 5),
    panel_size=(6, 3),
    print_max_auc=True,
):
    """Four-panel run summary: loss, learning rates, AUC, pose changes.

    Args:
        source: An ``EPO`` instance, a path to ``training_logs.json``, or a
            dict shaped like that JSON.
        auc_thresholds: AUC buckets (degrees) to overlay in the AUC panel.
        panel_size: ``(width, height)`` of a single subplot in inches; the
            figure is sized as ``(4 * width, height)``.
        print_max_auc: If True, print the max AUC@5 before plotting (matches
            the original notebook cell's status line).

    Returns:
        ``(fig, axes)`` where ``axes`` is the 4-element array from
        ``plt.subplots`` (loss, lr, auc, changes).
    """
    data = _extract(source)
    auc = data["auc"]
    if print_max_auc and auc.get(5):
        print(f"Max AUC@5: {max(auc[5])}")

    n = 4
    fig, axes = plt.subplots(1, n, figsize=(panel_size[0] * n, panel_size[1]))
    ax_loss, ax_lr, ax_auc, ax_changes = axes

    # Panel 1 — raw loss vs step index (matches the notebook's plt.plot(loss)).
    ax_loss.plot(data["raw_loss"], label="Loss")
    ax_loss.legend()

    # Panel 2 — per-group learning rates; each entry is [(step, lr), ...].
    for group, series in data["lr_list"].items():
        arr = np.asarray(series, dtype=float)
        if arr.size:
            ax_lr.plot(arr[:, 0], arr[:, 1], label=f"LR of {group}")
    if ax_lr.has_data():
        ax_lr.legend()

    # Panel 3 — AUC curves at the requested thresholds.
    auc_steps = data["auc_steps"]
    plotted_any = False
    for th in auc_thresholds:
        if auc.get(th):
            ax_auc.plot(auc_steps, auc[th], label=f"AUC@{th}px")
            plotted_any = True
    if plotted_any:
        ax_auc.legend()

    # Panel 4 — rotation / translation / max change vs step.
    steps = data["steps"]
    if data["q_changes"].size:
        ax_changes.plot(steps, data["q_changes"], label="Rotation change (deg)")
    if data["t_changes"].size:
        ax_changes.plot(steps, data["t_changes"], label="Translation change (deg)")
    if data["max_changes"].size:
        ax_changes.plot(steps, data["max_changes"], label="Max change")
    ax_changes.set_xlabel("Optimization steps")
    ax_changes.set_ylabel("Change")
    if ax_changes.has_data():
        ax_changes.legend()

    fig.tight_layout()
    return fig, axes


def _flatten_residuals(residuals):
    """Flatten ``{(i, j): [(step, val), ...]}`` into parallel numpy arrays.

    Returns ``(pair_keys, pair_idx, steps, vals)`` where ``pair_idx[k]`` is
    the row index in ``pair_keys`` for the k-th (step, val) entry.
    """
    pair_keys = list(residuals.keys())
    if not pair_keys:
        return [], np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)

    idx_parts, step_parts, val_parts = [], [], []
    for row, key in enumerate(pair_keys):
        entries = residuals[key]
        if not entries:
            continue
        arr = np.asarray(entries, dtype=np.float64)
        n = arr.shape[0]
        idx_parts.append(np.full(n, row, dtype=np.int64))
        step_parts.append(arr[:, 0].astype(np.int64))
        val_parts.append(arr[:, 1].astype(np.float32))

    if not idx_parts:
        return (
            pair_keys,
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty(0, np.float32),
        )

    return (
        pair_keys,
        np.concatenate(idx_parts),
        np.concatenate(step_parts),
        np.concatenate(val_parts),
    )


def _per_pair_endpoint_means(pair_idx, all_steps, all_vals, n_pairs, window):
    """Return per-pair (first-window mean, last-window mean, n_entries).

    Sorts entries by ``(pair_idx, step)`` once, then slices each pair's
    contiguous chunk to compute the leading/trailing window means.
    Pairs with fewer than ``2 * window`` entries get NaN — too little
    signal to call them lagging vs. converged.
    """
    order = np.lexsort((all_steps, pair_idx))
    s_pair = pair_idx[order]
    s_vals = all_vals[order]
    boundaries = np.searchsorted(s_pair, np.arange(n_pairs + 1))

    init_mean = np.full(n_pairs, np.nan, dtype=np.float32)
    final_mean = np.full(n_pairs, np.nan, dtype=np.float32)
    n_entries = np.zeros(n_pairs, dtype=np.int64)
    for r in range(n_pairs):
        s, e = boundaries[r], boundaries[r + 1]
        n = e - s
        n_entries[r] = n
        if n >= 2 * window:
            init_mean[r] = s_vals[s : s + window].mean()
            final_mean[r] = s_vals[e - window : e].mean()
    return init_mean, final_mean, n_entries


def find_lagging_pairs(source, *, top_k=20, window=10, eps=1e-6):
    """Rank pairs that failed to follow the cohort's error reduction.

    For each pair, compares its mean loss over the first ``window``
    recorded entries to its mean over the last ``window``:

        improvement = (init_mean - final_mean) / max(init_mean, eps)

    Improvement near 0 (or negative) means the pair didn't converge —
    or its loss went up. Pairs with too few entries
    (``< 2 * window``) are skipped.

    Args:
        source: ``EPO`` instance with ``.residuals``, or the dict.
        top_k: Return at most this many laggers.
        window: Window size (steps) for the leading/trailing means.
        eps: Floor on ``init_mean`` to avoid div-by-zero.

    Returns:
        List of dicts sorted ascending by ``improvement``
        (worst-converging first), each with keys ``pair``,
        ``improvement``, ``init_mean``, ``final_mean``, ``n_entries``.
    """
    residuals = source.residuals if hasattr(source, "residuals") else source
    if not residuals:
        return []

    pair_keys, pair_idx, all_steps, all_vals = _flatten_residuals(residuals)
    if all_steps.size == 0:
        return []
    n_pairs = len(pair_keys)
    init_mean, final_mean, n_entries = _per_pair_endpoint_means(
        pair_idx, all_steps, all_vals, n_pairs, window
    )
    improvement = (init_mean - final_mean) / np.maximum(init_mean, eps)
    # Pairs with NaN improvement (insufficient entries) are excluded.
    valid = np.isfinite(improvement)
    valid_rows = np.where(valid)[0]
    order = valid_rows[np.argsort(improvement[valid_rows])]
    take = order[: max(0, int(top_k))]
    return [
        {
            "pair": pair_keys[r],
            "improvement": float(improvement[r]),
            "init_mean": float(init_mean[r]),
            "final_mean": float(final_mean[r]),
            "n_entries": int(n_entries[r]),
        }
        for r in take
    ]


def plot_per_pair_losses(
    source,
    *,
    log_y=True,
    figsize=(10, 5),
    line_alpha=0.85,
    max_lines=20,
    select="spread",
    cmap="viridis",
    show_legend=None,
    title="Per-pair loss trajectories",
):
    """Plot individual viewgraph-pair loss trajectories vs. step.

    Reads ``EPO.residuals`` — a dict
    ``{(i, j): [(step, residual), ...]}`` populated when ``debug=True``
    is passed to ``EPO.__call__``. A pair that was not drawn at step
    ``t`` simply has no entry and is rendered as a gap in its line.

    Args:
        source: An ``EPO`` instance with a ``.residuals`` attribute, or
            the dict itself.
        log_y: Use a symlog y-axis so outlier pairs don't squash the bulk.
        figsize: Forwarded to ``plt.subplots``.
        line_alpha: Per-pair line alpha.
        max_lines: Number of pair trajectories to draw.
        select: Which ``max_lines`` pairs to draw out of all recorded.
            Pairs are first ranked, then sampled:

            * ``"spread"`` — rank by mean loss over the run, then take
              ``max_lines`` points evenly spaced through the ranking.
              Shows best, median, and worst on one plot.
            * ``"worst"`` — same mean-loss ranking, take only the
              top-``max_lines`` (highest mean loss).
            * ``"lagging"`` — rank by failure to converge:
              ``(init_window_mean − final_window_mean) / init_window_mean``.
              Pairs near 0 didn't reduce; < 0 got worse. Takes the
              ``max_lines`` worst by this metric, independent of their
              absolute loss level. Use this to find pairs not following
              the cohort's reduction.
            * ``"uniform"`` — no ranking; ``max_lines`` pairs in dict
              insertion order.
        cmap: Colormap used to color the lines. Colors track the chosen
            ranking so outliers stand out at a glance.
        show_legend: Show a legend listing the pair keys. ``None``
            auto-shows when ``max_lines <= 20``.
        title: Axes title.

    Returns:
        ``(fig, ax)``.
    """
    residuals = source.residuals if hasattr(source, "residuals") else source
    if not residuals:
        raise ValueError(
            "No per-pair residuals recorded. Pass debug=True to "
            "EPO.__call__ so compute_batched_loss populates self.residuals."
        )

    pair_keys, pair_idx, all_steps, all_vals = _flatten_residuals(residuals)
    n_pairs_total = len(pair_keys)
    if all_steps.size == 0:
        raise ValueError("All recorded pair trajectories are empty.")

    n_steps = int(all_steps.max()) + 1
    steps = np.arange(n_steps)

    # Per-pair mean loss — used by select="spread"/"worst" and by the
    # worst-10 summary printed at the end (always).
    mean_per_pair = np.bincount(
        pair_idx, weights=all_vals, minlength=n_pairs_total
    ) / np.maximum(np.bincount(pair_idx, minlength=n_pairs_total), 1)
    mean_order = np.argsort(mean_per_pair)  # ascending: low loss → high loss

    # Pick which pairs to draw.
    if max_lines is None or max_lines >= n_pairs_total:
        n_draw = n_pairs_total
    else:
        n_draw = max(1, int(max_lines))

    if select == "uniform":
        draw_rows = np.linspace(0, n_pairs_total - 1, n_draw).astype(np.int64)
    elif select == "lagging":
        # Rank pairs by how little their loss reduced over the run.
        init_mean, final_mean, _ = _per_pair_endpoint_means(
            pair_idx, all_steps, all_vals, n_pairs_total, window=10
        )
        improvement = (init_mean - final_mean) / np.maximum(init_mean, 1e-6)
        valid = np.where(np.isfinite(improvement))[0]
        if valid.size == 0:
            raise ValueError("No pairs with enough history for select='lagging'.")
        order = valid[np.argsort(improvement[valid])]  # worst first
        draw_rows = order[:n_draw]
    elif select == "worst":
        draw_rows = mean_order[-n_draw:][::-1]  # worst first
    elif select == "spread":
        picks = np.linspace(0, n_pairs_total - 1, n_draw).astype(np.int64)
        draw_rows = mean_order[picks]
    else:
        raise ValueError(
            f"Unknown select={select!r}; use 'spread', 'worst', "
            "'lagging', or 'uniform'."
        )

    # Build the (n_draw, n_steps) matrix in one masked scatter from the flat arrays.
    rows_matrix = np.full((n_draw, n_steps), np.nan, dtype=np.float32)
    row_to_bg = np.full(n_pairs_total, -1, dtype=np.int64)
    row_to_bg[draw_rows] = np.arange(n_draw)
    bg_idx = row_to_bg[pair_idx]
    keep = bg_idx >= 0
    rows_matrix[bg_idx[keep], all_steps[keep]] = all_vals[keep]

    # Colors: walk the colormap in pick-order, so neighbours in the chosen
    # ranking get neighbouring colors.
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(np.linspace(0, 1, n_draw))

    # Cohort mean per step over ALL recorded pairs — the herd reference
    # to compare the drawn trajectories against. Always shown in red.
    sums_step = np.bincount(all_steps, weights=all_vals, minlength=n_steps)
    counts_step = np.bincount(all_steps, minlength=n_steps)
    with np.errstate(invalid="ignore", divide="ignore"):
        cohort_mean = np.where(counts_step > 0, sums_step / counts_step, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    segs = np.empty((n_draw, n_steps, 2), dtype=float)
    segs[:, :, 0] = steps
    segs[:, :, 1] = rows_matrix
    lc = LineCollection(
        segs,
        colors=colors,
        linewidths=1.2,
        alpha=line_alpha,
    )
    ax.add_collection(lc)
    ax.plot(
        steps,
        cohort_mean,
        color="red",
        linewidth=2.0,
        label="cohort mean (all pairs)",
        zorder=5,
    )

    ax.set_xlim(steps.min(), steps.max())
    with np.errstate(all="ignore"):
        ymin = float(np.nanmin(rows_matrix))
        ymax = float(np.nanmax(rows_matrix))
    if np.isfinite(ymin) and np.isfinite(ymax):
        ax.set_ylim(ymin, ymax)

    if log_y:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Pair loss")
    ax.set_title(
        f"{title}  ({n_draw}/{n_pairs_total} pairs by {select}, {n_steps} steps)"
    )
    ax.grid(axis="y", alpha=0.3)

    if show_legend is None:
        show_legend = n_draw <= 20
    if show_legend:
        # Build proxy Line2D handles so the legend matches LineCollection colors.
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color="red", linewidth=2.0, label="cohort mean"),
        ] + [
            Line2D([0], [0], color=colors[k], label=str(pair_keys[draw_rows[k]]))
            for k in range(n_draw)
        ]
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="x-small",
            frameon=False,
        )

    # Print the worst-10 pairs as ``name_i  name_j`` per line, in
    # descending-mean-loss order. Copy-paste into a pair blacklist for a
    # follow-up run.
    idx_to_name = None
    if hasattr(source, "poses") and hasattr(source.poses, "tensor_idx_to_image"):
        idx_to_name = source.poses.tensor_idx_to_image

    n_print = min(100, n_pairs_total)
    worst_rows = mean_order[-n_print:][::-1]
    for r in worst_rows:
        i, j = pair_keys[r]
        names = []
        for idx in (i, j):
            name = None
            if idx_to_name is not None:
                try:
                    name = idx_to_name[idx]
                except (KeyError, IndexError, TypeError):
                    name = None
            names.append(name if name is not None else str(idx))
        print(f"{names[0]}  {names[1]}")

    fig.tight_layout()
    return fig, ax


def plot_per_image_losses(
    source,
    *,
    log_y=True,
    figsize=(10, 5),
    line_alpha=0.4,
    title="Per-image loss trajectories",
):
    """Aggregate per-pair losses to per-image and plot one line per image.

    Each pair ``(i, j)`` contributes its loss to both image ``i`` and
    image ``j``. An image's value at step ``t`` is the mean of pair
    losses touching it at that step. Output shape is
    ``(n_images, n_steps)`` — small enough that the full matrix and its
    percentile envelopes are trivial.

    Reads ``EPO.residuals`` (``{(i, j): [(step, residual), ...]}``),
    populated when ``debug=True`` is passed to ``EPO.__call__``.

    Returns:
        ``(fig, ax)``.
    """
    residuals = source.residuals if hasattr(source, "residuals") else source
    if not residuals:
        raise ValueError(
            "No per-pair residuals recorded. Pass debug=True to "
            "EPO.__call__ so compute_batched_loss populates self.residuals."
        )

    pair_keys, pair_idx, all_steps, all_vals = _flatten_residuals(residuals)
    if all_steps.size == 0:
        raise ValueError("All recorded pair trajectories are empty.")

    n_steps = int(all_steps.max()) + 1

    image_ids_set = set()
    for i, j in pair_keys:
        image_ids_set.add(i)
        image_ids_set.add(j)
    image_ids = sorted(image_ids_set)
    id_to_row = {iid: r for r, iid in enumerate(image_ids)}
    n_imgs = len(image_ids)

    # Map each flat entry's pair_idx to (image_i_row, image_j_row) via a
    # lookup table built once. Then do the entire per-(image, step)
    # reduction in two bincount calls over linear indices.
    i_lookup = np.fromiter(
        (id_to_row[pair_keys[r][0]] for r in range(len(pair_keys))),
        dtype=np.int64,
        count=len(pair_keys),
    )
    j_lookup = np.fromiter(
        (id_to_row[pair_keys[r][1]] for r in range(len(pair_keys))),
        dtype=np.int64,
        count=len(pair_keys),
    )
    all_i = i_lookup[pair_idx]
    all_j = j_lookup[pair_idx]

    total_cells = n_imgs * n_steps
    lin_i = all_i * n_steps + all_steps
    lin_j = all_j * n_steps + all_steps
    sums = np.bincount(lin_i, weights=all_vals, minlength=total_cells)
    sums += np.bincount(lin_j, weights=all_vals, minlength=total_cells)
    counts = np.bincount(lin_i, minlength=total_cells)
    counts += np.bincount(lin_j, minlength=total_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(counts > 0, sums / counts, np.nan).astype(np.float32)
    matrix = matrix.reshape(n_imgs, n_steps)

    steps = np.arange(n_steps)
    fig, ax = plt.subplots(figsize=figsize)

    # Single LineCollection: one artist for all n_imgs lines.
    segs = np.empty((n_imgs, n_steps, 2), dtype=float)
    segs[:, :, 0] = steps
    segs[:, :, 1] = matrix
    lc = LineCollection(
        segs,
        colors="steelblue",
        linewidths=0.8,
        alpha=line_alpha,
    )
    ax.add_collection(lc)

    with np.errstate(all="ignore"):
        p50 = np.nanmedian(matrix, axis=0)
        p90 = np.nanpercentile(matrix, 90, axis=0)
        pmax = np.nanmax(matrix, axis=0)
    ax.plot(steps, p50, color="black", linewidth=1.5, label="median")
    ax.plot(steps, p90, color="darkorange", linewidth=1.2, label="p90")
    ax.plot(steps, pmax, color="crimson", linewidth=1.0, label="max", alpha=0.8)

    ax.set_xlim(steps.min(), steps.max())
    ymin = float(np.nanmin(matrix))
    ymax = float(np.nanmax(matrix))
    if np.isfinite(ymin) and np.isfinite(ymax):
        ax.set_ylim(ymin, ymax)

    if log_y:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Mean pair loss touching image")
    ax.set_title(f"{title}  ({n_imgs} images, {n_steps} steps)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig, ax
