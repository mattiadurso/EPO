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
from helpers.reprojection_compiled import project_world_to_2D
from helpers.reconstruction import build_reconstruction
from matplotlib.colors import Normalize

from scipy.spatial.transform import Rotation as R


class ReconstructAndVizModule:
    """Module for reconstruction export and visualization functions. Just to have less code in Adjuster."""

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
        output_path="optimized_reconstruction_GD",
        save_points=True,
        verbose=False,
        max_points_per_image=100_000,
        final_dbscan_filtering=False,
        dbscan_eps=0.05,
        dbscan_min_samples=5,
        gt_path=None,
    ):
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

        # save loading time and optimization time in timings.txt in same folder as output_path
        timings_path = os.path.join(output_path, "timings.txt")
        with open(timings_path, "w") as f:
            for key, value in self.timings.items():
                if "total" not in key:
                    f.write(f"{key}: {value:.4f} s\n")
            # total at the end
            total = 0.0
            for key, value in self.timings.items():
                if "total" in key:
                    total += value
                    f.write(f"{key}: {value}\n")
            f.write(f"total: {total}\n")

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

    def get_frustum_strips(self, scale=0.15, w=1.0, h=1.0):
        # Normalize aspect to fit in scale
        max_dim = max(w, h)
        w_sc = (w / max_dim) * scale * 0.5
        h_sc = (h / max_dim) * scale * 0.5
        z = scale

        # Corners in Camera coords (Right, Down, Forward) -> (+X, +Y, +Z)
        tr = [w_sc, h_sc, z]
        br = [w_sc, -h_sc, z]
        bl = [-w_sc, -h_sc, z]
        tl = [-w_sc, h_sc, z]
        o = [0, 0, 0]

        return [
            [tr, br, bl, tl, tr],  # Image plane
            [o, tr],
            [o, br],
            [o, bl],
            [o, tl],  # Ray to corners
        ]

    def log_reconstruction_rerun(
        self,
        path,
        entity,
        static_cameras=False,
        points3D=False,
        static_points=False,
        camera_color=[0, 255, 0],
    ):
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
            strips = self.get_frustum_strips(scale=0.15, w=cam.width, h=cam.height)
            rr.log(
                f"world/{entity}/{img.name}/cam",
                rr.LineStrips3D(strips, colors=camera_color, radii=0.002),
                static=static_cameras,
            )

        # Log GT Point Cloud
        if len(recon.points3D) > 0 and points3D:
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



    #TODO: single image update
    def to_TUM(self, image_name_to_ts, output_path="TUM_RGBD.txt", images_list=[]):
        image_names = self.image_id_map.keys() if len(images_list) == 0 else images_list
        M = np.diag([1, -1, -1])

        existing_ts = set()
        try:
            with open(output_path, "r") as f_check:
                for line in f_check:
                    if line.strip():
                        existing_ts.add(line.split(" ")[0])
        except FileNotFoundError:
            pass

        with open(output_path, "a") as f_out:
            for image in sorted(image_names):
                ts = image_name_to_ts.get(image)
                if ts is None:
                    continue
                ts = str(ts)

                if ts in existing_ts:
                    continue

                idx = self.image_id_map[image]

                # Read directly from the cached 4x4 matrix — correct regardless
                # of whether this frame was optimized via forward_frame (no MLP)
                # or the global loop (with MLP). Both paths write to self.poses.poses.
                P_w2c = self.poses.poses[idx].detach().cpu().numpy()  # (4, 4)
                R_w2c = P_w2c[:3, :3]
                t_w2c = P_w2c[:3, 3]

                R_c2w = R_w2c.T
                t_c2w = -R_w2c.T @ t_w2c

                t_tum = M @ t_c2w
                R_tum = M @ R_c2w

                q_tum = R.from_matrix(R_tum).as_quat()

                out_line = (f"{ts} {t_tum[0]:.6f} {t_tum[1]:.6f} {t_tum[2]:.6f} "
                            f"{q_tum[0]:.6f} {q_tum[1]:.6f} {q_tum[2]:.6f} {q_tum[3]:.6f}\n")
                f_out.write(out_line)
                f_out.flush()