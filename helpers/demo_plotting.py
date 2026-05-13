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
    """Draw a vertical line at every step where the convergence flag rises."""
    flags = [
        check_convergence_pure(values[: i + 1], mode, window, tol)
        for i in range(len(values))
    ]
    first = True
    for idx in _rising_edges(flags):
        ax.vlines(
            steps[idx],
            0,
            ymax,
            color=color,
            linestyle="--",
            linewidth=2,
            zorder=4,
            label=label if first else None,
        )
        first = False


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
        ax_changes.plot(steps, data["t_changes"], label="Translation change (m)")
    if data["max_changes"].size:
        ax_changes.plot(steps, data["max_changes"], label="Max change")
    ax_changes.set_xlabel("Optimization steps")
    ax_changes.set_ylabel("Change")
    if ax_changes.has_data():
        ax_changes.legend()

    fig.tight_layout()
    return fig, axes
