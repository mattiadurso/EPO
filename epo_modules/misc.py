import os
import torch
import random
import numpy as np


class MiscModule:
    """Miscellaneous module for utility functions. Just to have less code in epo."""

    def __init__(self):
        self.name = "Miscellaneous Module"

    def print_summary(self, w=None):
        """
        Print a summary of the timings and loss.
        Note:
            Total optimization might differer from parts sum by 0.2-0.3 seconds.
        """
        # Column widths
        key_width = 30
        val_width = 10
        perc_width = 8
        avg_width = 12

        # Total line width
        if w is None:
            w = key_width + val_width + perc_width + avg_width + 6

        print("\n" + "=" * w)
        print(f"{'Summary':^{w}}")
        print("-" * w)

        # Compute total
        self.timings["total"] = self.timings.get("total_loading", 0) + self.timings.get(
            "total_optimization", 0
        )

        # Header row
        print(
            f"{'Stage':<{key_width}}"
            f"{'Time (s)':>{val_width+2}}"
            f"{'%':>{perc_width-1}}"
            f"{'AVG/Iter (s)':>{avg_width+4}}"
        )
        print("-" * w)

        num_iters = len(getattr(self, "loss_list", []))

        ordered_keys = [
            "total_loading",
            "total_optimization",
            "step_pre_computation",
            "prepare_batched_inputs",
            "forward_pass",
            "loss_computation",
            "gradients_computation",
            "parameters_update",
            "logging",
            "early_stop_check",
        ]

        per_iter_keys = {
            "step_pre_computation",
            "prepare_batched_inputs",
            "forward_pass",
            "loss_computation",
            "gradients_computation",
            "parameters_update",
            "logging",
            "early_stop_check",
        }

        for key in ordered_keys:
            if key not in self.timings:
                continue

            value = self.timings[key]

            if value == 0 and key not in per_iter_keys:
                continue

            # Determine display key (indent non-total keys)
            display_key = key if "total" in key else "    " + key

            if key in per_iter_keys and num_iters > 0:
                # Show total time, percentage, AND per-iteration average
                perc = (
                    (value / self.timings["total"]) * 100
                    if self.timings["total"] > 0
                    else 0
                )
                value_avg = value / num_iters
                row_str = (
                    f"{display_key:<{key_width}}"
                    f"{value:>{val_width}.2f}"
                    f"{perc:>{perc_width}.1f}%"
                    f"{value_avg:>{avg_width}.4f}"
                )
            elif key == "total_optimization" and num_iters > 0:
                # Show total optimization time without per-iteration average
                perc = (
                    (value / self.timings["total"]) * 100
                    if self.timings["total"] > 0
                    else 0
                )
                row_str = (
                    f"{display_key:<{key_width}}"
                    f"{value:>{val_width}.2f}"
                    f"{perc:>{perc_width}.1f}%"
                    f"{'':>{avg_width}}"
                )
            else:
                # Show total time and percentage
                perc = (
                    (value / self.timings["total"]) * 100
                    if self.timings["total"] > 0
                    else 0
                )
                row_str = (
                    f"{display_key:<{key_width}}"
                    f"{value:>{val_width}.2f}"
                    f"{perc:>{perc_width}.1f}%"
                    f"{'':>{avg_width}}"
                )

            print(row_str)

        print("-" * w)
        total_avg = num_iters / self.timings["total_optimization"]
        print(
            f"{'Total':<{key_width}}"
            f"{self.timings['total']:>{val_width}.2f}"
            f"{'':>{perc_width}}"
            f"{total_avg:>{avg_width-1}.2f} it/s"
        )

        # Loss summary
        if len(self.loss_list) > 0:
            initial_loss = self.loss_list[0]
            final_loss = self.loss_list[-1]
            delta = initial_loss - final_loss

            print("-" * w)
            print(f"{'Initial loss:':<{key_width}}{initial_loss:>{val_width}.6f}")
            print(f"{'Final loss:':<{key_width}}{final_loss:>{val_width}.6f}")
            # loss and percentage with sign under % column on same row
            perc_improvement = (
                (delta / initial_loss) * 100 if initial_loss != 0 else 0.0
            )
            sign = "-" if perc_improvement > 0 else "+"
            perc_improvement = sign + f"{abs(perc_improvement):.1f}"
            print(
                f"{'Loss reduction:':<{key_width}}{delta:>{val_width}.6f}"
                f"{perc_improvement:>{perc_width}}%"
            )
            steps = len(self.loss_list)
            print(
                f"{'Mean (Edges) Reproj. Error:':<{key_width}}{self.mre.mean():>{val_width}.3f}"
            )
            print(f"{f'Total steps:':<{key_width}}{steps:>{val_width}d}")

        print("=" * w)

    def fix_seed(self, mode="inference"):
        assert mode in ["inference", "debug"]

        # 1. Standard Python/Numpy seeds
        random.seed(self.seed)
        np.random.seed(self.seed)

        # 2. PyTorch seeds
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # For multi-GPU

        # debug
        if mode == "debug":
            os.environ["PYTHONHASHSEED"] = "0"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging only - slow!
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        # inference
        if mode == "inference":
            # faster but non-deterministic, which introduce a bit of variability in the results
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

    def __repr__(self):
        repr_str = f"epo(\n"
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

        repr_str += f")"
        return repr_str

    ## Optimizer and scheduler
    def _collect_parameters_to_optimize(self):
        params_to_optimize = {}

        # Collect parameters to optimize
        params_to_optimize["q"] = self.poses.parameters(q=True)

        params_to_optimize["t"] = self.poses.parameters(t=True)

        params_to_optimize["mlp"] = self.poses.parameters(mlp=True)

        params_to_optimize["k"] = self.intrinsics.parameters()

        params_to_optimize["z"] = self.sampled_depth.parameters()

        return params_to_optimize

    def _print_params_summary(self, params_to_optimize):
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
