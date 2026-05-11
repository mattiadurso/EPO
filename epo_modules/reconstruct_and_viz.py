"""Mixin: reconstruction export, residual visualization, Rerun logging.

Bound onto :class:`epo.EPO`; functions assume ``self`` is an EPO instance
(uses ``self.images``, ``self.viewgraph``, ``self.verbose`` and the
learnable submodules).
"""

import os
import json
import torch
import random
import pycolmap
import rerun as rr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm
from losses.dt_loss import sample_distance_field
from helpers.reprojection import project_world_to_2D
from helpers.reconstruction import build_reconstruction
from matplotlib.colors import Normalize


class ReconstructAndVizModule:
    """Mixin: reconstruction export and visualization helpers for :class:`epo.EPO`."""

    @torch.no_grad()
    def visualize_residuals(
        self,
        output_dir="residual_maps",
        percentile=99,
        max_images=100,
        custom_viewgraph=None,
    ):
        """
        Visualize reprojection residuals for all image pairs in the viewgraph.
        Creates error maps showing where edges align well or poorly between image pairs.

        Args:
            output_dir (str): Directory to save residual visualization maps
            percentile (float): Percentile for colormap scaling (default 95 to avoid outliers)
        """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        os.makedirs(output_dir, exist_ok=True)

        # Select viewgraph pairs
        if custom_viewgraph:
            viewgraph = custom_viewgraph
        else:
            viewgraph = (
                self.viewgraph
                if max_images < 0
                else random.choices(self.viewgraph, k=max_images)
            )
        num_pairs = len(viewgraph)
        if self.verbose:
            print(f"Visualizing residuals for {num_pairs:,} image pairs...")

        for pair_idx, (img_i, img_j) in enumerate(
            tqdm(viewgraph, desc="Computing residuals")
        ):
            sampled_vg = [(img_i, img_j)]
            batch, pad_masks, dt_fields = self.create_batched_inputs(sampled_vg)
            # Project edges and compute residuals
            edges_reprojected, _ = project_world_to_2D(**batch)
            residuals = sample_distance_field(dt_fields, edges_reprojected).squeeze(1)

            # Forward: i->j
            res_ij = residuals[0]  # (N,)
            edges_ij = edges_reprojected[0]  # (N, 2)
            pad_mask_j = self.images[img_j]["pad_mask"]
            img_j_tensor = (
                self.images[img_j]["image"]
                if "image" in self.images[img_j]
                else torch.zeros(3, *self.images[img_j]["hw"])
            )
            edges_map_j = self.images[img_j]["edges_map"].cpu().numpy()
            hw_j = self.images[img_j]["hw"]

            # Backward: j->i
            res_ji = residuals[1]
            edges_ji = edges_reprojected[1]
            pad_mask_i = self.images[img_i]["pad_mask"]
            img_i_tensor = (
                self.images[img_i]["image"]
                if "image" in self.images[img_i]
                else torch.zeros(3, *self.images[img_i]["hw"])
            )
            edges_map_i = self.images[img_i]["edges_map"].cpu().numpy()
            hw_i = self.images[img_i]["hw"]

            # Save visualization
            self._save_residual_visualization_custom(
                img_i_tensor,
                edges_map_i,
                img_j_tensor,
                edges_map_j,
                edges_ij,
                res_ij,
                pad_mask_j,
                hw_j,
                edges_ji,
                res_ji,
                pad_mask_i,
                hw_i,
                img_i,
                img_j,
                output_dir,
                pair_idx,
                percentile,
            )

    @torch.no_grad()
    def _save_residual_visualization_custom(
        self,
        img_i_tensor,
        edges_map_i,
        img_j_tensor,
        edges_map_j,
        edges_ij,
        res_ij,
        pad_mask_j,
        hw_j,
        edges_ji,
        res_ji,
        pad_mask_i,
        hw_i,
        img_i,
        img_j,
        output_dir,
        pair_idx,
        percentile,
    ):
        """Render a side-by-side residual figure for a single (i, j) pair.

        Saves a PNG showing the two images, the projected edges colored by
        residual, and the underlying edge maps. Used by
        :meth:`visualize_residuals` to dump per-pair diagnostics.
        """
        # Prepare filenames
        safe_img_i = img_i.replace("/", "_").replace("\\", "_")
        safe_img_j = img_j.replace("/", "_").replace("\\", "_")
        filename = f"{pair_idx:04d}_{safe_img_i}_to_{safe_img_j}.png"
        filepath = os.path.join(output_dir, filename)

        # Normalize images
        img_i_np = img_i_tensor.cpu().numpy().transpose(1, 2, 0)
        img_j_np = img_j_tensor.cpu().numpy().transpose(1, 2, 0)
        img_i_np = np.clip(
            img_i_np / (img_i_np.max() if img_i_np.max() > 1.0 else 1.0), 0, 1
        )
        img_j_np = np.clip(
            img_j_np / (img_j_np.max() if img_j_np.max() > 1.0 else 1.0), 0, 1
        )

        # Edge maps: white edges on black
        edges_img_i = np.zeros((*hw_i, 3), dtype=np.float32)
        edges_img_i[edges_map_i > 0] = 1.0
        edges_img_j = np.zeros((*hw_j, 3), dtype=np.float32)
        edges_img_j[edges_map_j > 0] = 1.0

        # Row 1, Col 3: image i edges (white) + projected edges from j (colored)
        combined_i = np.zeros((*hw_i, 3), dtype=np.float32)
        combined_i[edges_map_i > 0] = 1.0  # white edges from i
        valid_mask = pad_mask_i > 0.5
        valid_edges = edges_ji[valid_mask].long().cpu().numpy()
        valid_residuals = res_ji[valid_mask].cpu().numpy()
        if valid_edges.shape[0] > 0:
            vmax = np.percentile(valid_residuals, percentile)
            norm = Normalize(vmin=0, vmax=vmax)
            cmap = cm.get_cmap("RdYlGn_r")
            for idx, (x, y) in enumerate(valid_edges):
                color = cmap(norm(valid_residuals[idx]))[:3]
                x = np.clip(x, 0, hw_i[1] - 1)
                y = np.clip(y, 0, hw_i[0] - 1)
                combined_i[y, x] = color

        # Row 2, Col 3: image j edges (white) + projected edges from i (colored)
        combined_j = np.zeros((*hw_j, 3), dtype=np.float32)
        combined_j[edges_map_j > 0] = 1.0  # white edges from j
        valid_mask = pad_mask_j > 0.5
        valid_edges = edges_ij[valid_mask].long().cpu().numpy()
        valid_residuals = res_ij[valid_mask].cpu().numpy()
        if valid_edges.shape[0] > 0:
            vmax = np.percentile(valid_residuals, percentile)
            norm = Normalize(vmin=0, vmax=vmax)
            cmap = cm.get_cmap("RdYlGn_r")
            for idx, (x, y) in enumerate(valid_edges):
                color = cmap(norm(valid_residuals[idx]))[:3]
                x = np.clip(x, 0, hw_j[1] - 1)
                y = np.clip(y, 0, hw_j[0] - 1)
                combined_j[y, x] = color

        # Compute mean residual for display (as in loss), clmap and apply huber loss
        # mean_residual = 0.5 * (res_ij.mean().item() + res_ji.mean().item())
        max_residual_ij = res_ij.max().item()
        max_residual_ji = res_ji.max().item()

        res_ij = res_ij.clamp(max=10.0)
        res_ji = res_ji.clamp(max=10.0)
        delta = 1.0
        huber_ij = (
            0.5 * res_ij**2 * (res_ij <= delta).float()
            + (delta * (res_ij - 0.5 * delta)) * (res_ij > delta).float()
        )
        huber_ji = (
            0.5 * res_ji**2 * (res_ji <= delta).float()
            + (delta * (res_ji - 0.5 * delta)) * (res_ji > delta).float()
        )
        mean_residual = 0.5 * (huber_ij.mean().item() + huber_ji.mean().item())

        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))  # Reduced size
        plt.subplots_adjust(wspace=0.08, hspace=0.08)  # Less space between columns/rows

        # Add residual value in top-left corner (black text)
        fig.text(
            0.12,
            0.9,
            f"Residual: {mean_residual:.3f}, Max residuals: {max_residual_ij:.3f} and {max_residual_ji:.3f}",
            ha="left",
            va="top",
            color="black",
            fontsize=10,
            weight="bold",
        )

        # Row 1: image i, edges i, edges i + projected edges from j
        axes[0, 0].imshow(img_i_np)
        axes[0, 0].set_title(f"Image: {img_i}")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(edges_img_i)
        axes[0, 1].set_title(f"Edges: {img_i}")
        axes[0, 1].axis("off")
        axes[0, 2].imshow(combined_i)
        axes[0, 2].set_title(f"Edges {img_i} + proj. edges from {img_j}")
        axes[0, 2].axis("off")

        # Row 2: image j, edges j, edges j + projected edges from i
        axes[1, 0].imshow(img_j_np)
        axes[1, 0].set_title(f"Image: {img_j}")
        axes[1, 0].axis("off")
        axes[1, 1].imshow(edges_img_j)
        axes[1, 1].set_title(f"Edges: {img_j}")
        axes[1, 1].axis("off")
        axes[1, 2].imshow(combined_j)
        axes[1, 2].set_title(f"Edges {img_j} + proj. edges from {img_i}")
        axes[1, 2].axis("off")

        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

    def to_colmap(
        self,
        output_path: str = "optimized_reconstruction_GD",
        save_points: bool = True,
        verbose: bool = False,
        max_points_per_image: int = 100_000,
        final_dbscan_filtering: bool = False,
        dbscan_eps: float = 0.05,
        dbscan_min_samples: int = 5,
        gt_path: str | None = None,
        save_depth: bool = False,
    ):
        """Export the optimized reconstruction (and optionally depth maps) to COLMAP format.

        Args:
            output_path: Destination folder. Existing contents are removed.
            save_points: If True, unproject edge points and write them as
                a sparse point cloud.
            verbose: If True, log per-step progress from the underlying
                build / DBSCAN passes.
            max_points_per_image: Cap on points exported per image.
            final_dbscan_filtering: If True, run a final DBSCAN pass that
                keeps only the largest cluster of 3D points.
            dbscan_eps, dbscan_min_samples: DBSCAN hyperparameters.
            gt_path: If given, align the exported reconstruction to this GT
                model with ``colmap model_aligner``.
            save_depth: If True, write per-image refined depth maps as ``.h5``.
        """
        os.system(f"rm -rf {output_path}/*")
        recon = build_reconstruction(
            self,
            output_path=output_path,
            save_points=save_points,
            verbose=verbose,
            max_points_per_image=max_points_per_image,
            final_dbscan_filtering=final_dbscan_filtering,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            bin=True,
        )

        if gt_path is not None:
            # align
            os.system(
                f"colmap model_aligner \
                    --input_path {output_path} \
                    --output_path {output_path} \
                    --ref_model_path {gt_path} \
                    --alignment_max_error 1 > /dev/null 2>&1"
            )

        if save_depth:
            if self.verbose:
                print("Saving depth maps...")
            # saving depth in same format as input
            import h5py

            os.makedirs(os.path.join(output_path, "depth"), exist_ok=True)
            for image_name, id in self.image_id_map.items():
                # read depth, edges, padding and images size
                depth = self.sampled_depth.get_parameters([id]).detach().cpu().squeeze()
                edges_pad = (
                    self.pad_masks.get_parameters([id]).detach().cpu().squeeze().float()
                )
                depth = depth * edges_pad  # N
                edges = (
                    self.edges_padded.get_parameters([id])
                    .detach()
                    .cpu()
                    .squeeze()
                    .long()
                )  # N,2

                hw = self.images[image_name]["hw"]
                hw_array = torch.zeros(hw, device="cpu")

                # populate hw_array at edges location with depth values
                hw_array[edges[:, 1], edges[:, 0]] = depth
                # create cam folder
                cam = image_name.split("/")[0]
                os.makedirs(os.path.join(output_path, "depth", cam), exist_ok=True)
                # saving depth
                image_name = image_name.split(".")[0] + ".h5"
                with h5py.File(
                    os.path.join(output_path, "depth", image_name), "w"
                ) as f:
                    f.create_dataset("depth", data=hw_array.numpy(), compression="gzip")

        # Keep `total` consistent whether or not `forward()` has been called
        # (e.g. if print_summary is skipped).  benchmark_plotting.py reads
        # the last line of timings.txt as `total: <s>`, and benchmark.ipynb
        # reads training_logs.json["timings"]["total"], so both files must
        # have this key.
        self.timings["total"] = (
            self.timings.get("total_loading", 0.0)
            + self.timings.get("total_optimization", 0.0)
        )

        timings_path = os.path.join(output_path, "timings.txt")
        with open(timings_path, "w") as f:
            # ── per-iteration accumulators + one-shot times ──────────
            skip = {"total_loading", "total_optimization", "total"}
            for key, value in self.timings.items():
                if key in skip:
                    continue
                try:
                    f.write(f"{key}: {value:.4f} s\n")
                except (TypeError, ValueError):
                    f.write(f"{key}: {value}\n")
            # ── totals (must end with `total:` for benchmark_plotting) ─
            f.write(f"total_loading: {self.timings.get('total_loading', 0.0):.4f} s\n")
            f.write(f"total_optimization: {self.timings.get('total_optimization', 0.0):.4f} s\n")
            f.write(f"total: {self.timings['total']:.4f}\n")

        # save training data such metrics series too as dict
        training_logs = {
            "steps_total": self.max_num_iterations,
            "steps_actual": self.completed_iterations,
            "list_loss": self.loss_list,
            "list_lr": self.lr_list,
            "auc_saving_freq": self.auc_saving_freq,
            "list_auc": self.auc_list,
            "list_changes": self.changes,
            "convergence_first": self.mlp_pose_convergence,
            "convergence_second": self.optim_convergence,
            "timings": self.timings,
            "max_edges": self.max_edges,
            "len_viewgraph": len(self.viewgraph),
            "window_pose": self.window_pose,
            "window_depth": self.window_depth,
            "convergence_tol_pose": self.convergence_tol_pose,
            "convergence_tol_depth": self.convergence_tol_depth,
            "min_viewgraph_points": self.min_points,
            "reprojection_error": self.reprojection_error,
            "sampling_factor": self.sampling_factor,
            "grad_t_offset": self.grad_t_offset,
            "grad_k": self.grad_k,
            "grad_z": self.grad_z,
            "grad_mlp_pose_refinement": self.use_mlp_pose_refinement,
            "lr_mlp_pose": self.mlp_pose_lr,
            "lr_k": self.k_lr,
            "lr_z": self.z_lr,
            "lr_R": self.R_lr,
            "lr_t": self.t_lr,
            "matcher_type": self.matcher_type,
        }
        if hasattr(self, "mre"):
            training_logs["observations"] = self.pad_masks.params.sum().item()
            training_logs["mean_reproj_error"] = self.mre.mean().item()
            training_logs["median_reproj_error"] = np.median(self.mre).item()
        # sort keys alphabetically
        training_logs = dict(sorted(training_logs.items()))
        training_logs_path = os.path.join(output_path, "training_logs.json")
        with open(training_logs_path, "w") as f:
            json.dump(training_logs, f, indent=4)

        return recon

    def get_frustum_strips(
        self, scale: float = 0.15, w: float = 1.0, h: float = 1.0
    ) -> list[list[list[float]]]:
        """Return Rerun ``LineStrips3D`` describing a camera frustum.

        Builds a unit pyramid in camera-local coordinates (right-down-forward)
        with apex at the origin and an image plane at depth ``scale``,
        rescaled to preserve the ``w:h`` aspect ratio.

        Returns:
            A list of 5 polylines: the image-plane rectangle plus the four
            rays from the apex to its corners.
        """
        # Normalize aspect to fit within `scale`
        max_dim = max(w, h)
        w_sc = (w / max_dim) * scale * 0.5
        h_sc = (h / max_dim) * scale * 0.5
        z = scale

        # Corners in camera coords (right-down-forward): top-right, bottom-right, etc.
        top_right = [w_sc, h_sc, z]
        bot_right = [w_sc, -h_sc, z]
        bot_left = [-w_sc, -h_sc, z]
        top_left = [-w_sc, h_sc, z]
        origin = [0, 0, 0]

        return [
            [top_right, bot_right, bot_left, top_left, top_right],  # image plane
            [origin, top_right],
            [origin, bot_right],
            [origin, bot_left],
            [origin, top_left],
        ]

    def log_reconstruction_rerun(
        self,
        path: str,
        entity: str,
        static_cameras: bool = False,
        points3D: bool = False,
        static_points: bool = False,
        camera_color: list[int] = [0, 255, 0],
    ) -> None:
        """Log a COLMAP reconstruction (cameras + optional point cloud) to Rerun.

        Args:
            path: Path to a COLMAP reconstruction folder.
            entity: Top-level Rerun entity prefix (e.g. ``"gt"``, ``"opt"``).
            static_cameras: If True, log camera transforms as time-static.
            points3D: If True, also log the 3D point cloud.
            static_points: If True, log points as time-static.
            camera_color: RGB color used for the frustum line strips.
        """
        recon = pycolmap.Reconstruction(path)
        for img_id, img in recon.images.items():
            # COLMAP stores world-to-cam (R, t)
            # Rerun needs cam-to-world for the transform
            R_gt = torch.from_numpy(img.cam_from_world.rotation.matrix())
            t_gt = torch.from_numpy(img.cam_from_world.translation)

            # C = -R^T * t
            cam_center = -R_gt.T @ t_gt
            cam_rot = R_gt.T  # Rotation from camera to world

            rr.log(
                f"world/{entity}/{img.name}",
                rr.Transform3D(
                    translation=cam_center.numpy(),
                    mat3x3=cam_rot.numpy(),
                ),
                static=static_cameras,
            )

            # Frustum visualization
            cam = recon.cameras[img.camera_id]
            strips = self.get_frustum_strips(scale=0.20, w=cam.width, h=cam.height)
            rr.log(
                f"world/{entity}/{img.name}/cam",
                rr.LineStrips3D(strips, colors=camera_color, radii=0.005),
                static=static_cameras,
            )

        # Log GT Point Cloud
        if len(recon.points3D) > 0 and points3D:
            if self.verbose:
                print(f"Logging {len(recon.points3D)} GT points...")
            pts = []
            colors = []
            for p in recon.points3D.values():
                pts.append(p.xyz)
                colors.append(p.color)

            rr.log(
                f"world/{entity}/points",
                rr.Points3D(np.array(pts), colors=np.array(colors), radii=0.01),
                static=static_points,
            )
