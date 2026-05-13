"""Miscellaneous helpers mixed into :class:`epo.EPO`.

Holds the seeding, parameter-summary and timing-report utilities. Kept
separate from the main ``EPO`` class purely to keep ``epo.py`` shorter.
"""

import os
import random

import numpy as np
import torch


def _fmt_seconds(s: float) -> str:
    """Return a compact human-readable duration string."""
    if s < 0:
        return f"{s:.3f}s"
    if s < 1.0:
        return f"{s * 1000:.1f}ms"
    if s < 60.0:
        return f"{s:.2f}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {sec:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {sec:.0f}s"


class MiscModule:
    """Mixin: seeding, CUDA-sync helper and run-summary printing for :class:`epo.EPO`."""

    def __init__(self):
        self.name = "Miscellaneous Module"

    # ------------------------------------------------------------------
    # CUDA sync
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        """Block until all pending CUDA kernels are done (no-op on CPU).

        Call this immediately before reading ``time.perf_counter()`` at any
        timer boundary that follows GPU work.  Without it, PyTorch's async
        dispatch means the clock only captures CPU launch overhead, not actual
        GPU runtime.
        """
        if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Summary printing
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a structured timing + loss summary of the last EPO run."""
        t = self.timings
        num_iters = len(getattr(self, "loss_list", []))

        # ── column geometry ───────────────────────────────────────────
        KW = 32  # key column
        VW = 10  # total-time column
        PW = 7  # percentage column (incl. %)
        AW = 14  # avg/iter column
        W = KW + VW + PW + AW + 2

        sep = "=" * W
        thin = "-" * W

        def row(label, secs, perc=None, avg=None, indent=0):
            pad = "  " * indent
            lbl = f"{pad}{label}"
            s_t = _fmt_seconds(secs)
            perc_s = f"{perc:5.1f}%" if perc is not None else ""
            avg_s = f"{_fmt_seconds(avg)}/it" if avg is not None else ""
            return f"{lbl:<{KW}}" f"{s_t:>{VW}}" f"{perc_s:>{PW}}" f"{avg_s:>{AW}}"

        def pct(val, base):
            return (val / base * 100) if base > 0 else 0.0

        print(f"\n{sep}")
        print(f"{'EPO  Summary':^{W}}")

        # ── Loading ──────────────────────────────────────────────────
        total_loading = t.get("total_loading", 0.0)
        total_opt = t.get("total_optimization", 0.0)
        grand_total = total_loading + total_opt

        print(thin)
        print(f"{'LOADING':^{W}}")
        print(thin)

        loading_sub_keys = [
            ("load_images", "images"),
            ("load_depth_maps", "depth maps"),
            ("load_poses_and_intrinsics", "poses & intrinsics"),
            ("extract_edges", "edges"),
            ("compute_distance_fields", "distance fields"),
            ("compute_viewgraph", "viewgraph"),
        ]
        sub_sum = 0.0
        for key, label in loading_sub_keys:
            v = t.get(key, 0.0)
            if v > 0:
                print(row(label, v, pct(v, total_loading), indent=1))
                sub_sum += v

        other_loading = total_loading - sub_sum
        if other_loading > 0.01:
            print(
                row(
                    "other (extractor load, GC…)",
                    other_loading,
                    pct(other_loading, total_loading),
                    indent=1,
                )
            )

        print(row("total loading", total_loading, pct(total_loading, grand_total)))

        # ── Optimization loop ─────────────────────────────────────────
        print(thin)
        print(f"{'OPTIMIZATION  LOOP':^{W}}")
        print(thin)

        # Header
        print(
            f"{'stage':<{KW}}"
            f"{'total':>{VW}}"
            f"{'% opt':>{PW}}"
            f"{'avg / iter':>{AW}}"
        )
        print(thin)

        per_iter_keys = [
            ("step_pre_computation", "pre-compute  (unproject)"),
            ("prepare_batched_inputs", "batch inputs"),
            ("forward_pass", "forward  (project+Huber)"),
            ("loss_computation", "loss aggregation"),
            ("gradients_computation", "backward"),
            ("parameters_update", "optimizer step"),
            ("logging", "logging  (+ AUC, rerun)"),
            ("early_stop_check", "convergence check"),
        ]
        accounted = 0.0
        for key, label in per_iter_keys:
            v = t.get(key, 0.0)
            avg = (v / num_iters) if num_iters > 0 else None
            print(row(label, v, pct(v, total_opt), avg, indent=1))
            accounted += v

        # Unaccounted gap (untimed overhead between stages, rerun if not
        # use_rerun, sync overhead etc.)
        unaccounted = total_opt - accounted
        if abs(unaccounted) > 0.001:
            print(
                row("unaccounted", unaccounted, pct(unaccounted, total_opt), indent=1)
            )

        print(thin)
        it_per_s = (num_iters / total_opt) if total_opt > 0 else 0.0
        print(
            f"{'total optimization':<{KW}}"
            f"{_fmt_seconds(total_opt):>{VW}}"
            f"{'':>{PW}}"
            f"{it_per_s:>{AW - 5}.2f} it/s"
        )

        # ── One-shot post-processing ──────────────────────────────────
        one_shot = []
        if t.get("mre", 0.0) > 0:
            one_shot.append(("MRE forward pass", t["mre"]))
        if t.get("setup_visualization", 0.0) > 0.01:
            one_shot.append(("viz setup (GT/BA load)", t["setup_visualization"]))

        if one_shot:
            print(thin)
            print(f"{'ONE-SHOT  (not per-iter)':^{W}}")
            print(thin)
            for label, v in one_shot:
                print(row(label, v, indent=1))

        # ── Convergence milestones ────────────────────────────────────
        milestones = []
        if "pose_convergence_time" in t:
            milestones.append(("pose convergence", t["pose_convergence_time"]))
        if "depth_convergence_time" in t:
            milestones.append(("full convergence", t["depth_convergence_time"]))

        if milestones:
            print(thin)
            print(f"{'CONVERGENCE  (from opt start)':^{W}}")
            print(thin)
            for label, v in milestones:
                print(row(label, v, indent=1))

        # ── Grand total ───────────────────────────────────────────────
        print(thin)
        print(
            f"{'TOTAL  (loading + optimization)':<{KW}}"
            f"{_fmt_seconds(grand_total):>{VW}}"
        )

        # ── Loss / quality ────────────────────────────────────────────
        if num_iters > 0:
            print(thin)
            print(f"{'RESULTS':^{W}}")
            print(thin)

            initial_loss = self.loss_list[0]
            final_loss = self.loss_list[-1]
            delta = initial_loss - final_loss
            perc_imp = (delta / initial_loss * 100) if initial_loss != 0 else 0.0
            sign = "-" if perc_imp >= 0 else "+"

            print(f"  {'initial loss':<{KW - 2}}{initial_loss:>{VW}.6f}")
            print(f"  {'final loss':<{KW - 2}}{final_loss:>{VW}.6f}")
            print(
                f"  {'loss reduction':<{KW - 2}}{delta:>{VW}.6f}"
                f"  {sign}{abs(perc_imp):.1f}%"
            )
            print(f"  {'total steps':<{KW - 2}}{num_iters:>{VW}d}")

            if hasattr(self, "mre") and self.mre is not None:
                print(
                    f"  {'mean reprojection error':<{KW - 2}}"
                    f"{float(np.mean(self.mre)):>{VW}.3f} px"
                )

        print(sep + "\n")

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def fix_seed(self, mode: str = "inference") -> None:
        """Seed Python/NumPy/PyTorch RNGs and configure cuDNN.

        Args:
            mode: ``"inference"`` selects fast, non-deterministic kernels
                (cudnn.benchmark = True). ``"debug"`` enforces full
                determinism at a meaningful performance cost.
        """
        assert mode in ["inference", "debug"]

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if mode == "debug":
            os.environ["PYTHONHASHSEED"] = "0"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        if mode == "inference":
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                if "4090" in gpu_name:
                    torch.set_float32_matmul_precision("high")
                else:
                    torch.set_float32_matmul_precision("highest")

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a multi-line summary of the EPO instance state."""
        repr_str = "epo(\n"
        repr_str += f"  Reconstruction path: {self.reconstruction_path}\n"
        repr_str += f"  Images path: {self.images_path}\n"
        repr_str += f"  Depths path: {self.depths_path}\n"
        repr_str += f"  Number of images: {len(self.images)}\n"
        if hasattr(self, "viewgraph"):
            repr_str += f"  Number of viewgraph edges: {len(self.viewgraph):,}\n"

        total_params = 0
        params_to_optimize = self._collect_parameters_to_optimize()
        for key in ["k", "t", "q", "z"]:
            if key in params_to_optimize:
                set_params = sum(p.numel() for p in params_to_optimize[key])
                total_params += set_params

        repr_str += f"  Total parameters to optimize: {total_params:,}\n"
        converged_str = " (converged)" if getattr(self, "convergence", False) else ""
        repr_str += (
            f"  Number of optimization steps: {len(self.loss_list)}{converged_str}\n"
        )

        if len(self.loss_list) >= 2:
            initial_loss = self.loss_list[0]
            final_loss = self.loss_list[-1]
            delta = initial_loss - final_loss
            perc_improvement = (
                (delta / initial_loss) * 100 if initial_loss != 0 else 0.0
            )
            repr_str += f"  Loss improvement: {delta:.3f} ({perc_improvement:.2f}%)\n"

        repr_str += ")"
        return repr_str

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def _collect_parameters_to_optimize(self) -> dict:
        """Group every learnable parameter tensor by submodule key (R/t/mlp/k/z)."""
        return {
            "R": self.poses.parameters(R=True),
            "t": self.poses.parameters(t=True),
            "mlp": self.poses.parameters(mlp=True),
            "k": self.intrinsics.parameters(),
            "z": self.sampled_depth.parameters(),
        }

    def _print_params_summary(self, params_to_optimize: dict) -> None:
        """Print the per-module parameter count + grand total."""
        total_params = 0
        print("\nTotal parameters to optimize:")
        for key in ["t", "q", "mlp", "k", "z"]:
            space = 14 if key == "mlp" else 16
            if key not in params_to_optimize:
                print(f"  {key}: {0:>{space},}")
                continue
            set_params = sum(p.numel() for p in params_to_optimize[key])
            print(f"  {key}: {set_params:>{space},}")
            total_params += set_params
        print("-" * 23)
        print(f"  {'Total':}: {total_params:>12,}\n")
