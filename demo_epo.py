"""End-to-end demo: VGGT reconstruction → EPO refinement.

Runs the VGGT wrapper on a folder of images, then feeds the resulting
COLMAP reconstruction + dense depths to EPO for pose/depth refinement.
Writes the refined reconstruction to ``<output_path>/sparse_epo`` and
prints EPO's own timing summary plus the total end-to-end runtime.

Example:
python demo_epo.py \
    --images_path ~/Desktop/datasets/mipnerf360/kitchen/images_150 \
    --output_path optimized_reconstruction/demo \
    --gt_path ~/Desktop/datasets/mipnerf360/kitchen/sparse_150

Pass ``--vggt_output <dir>`` to skip VGGT and reuse a previous run's
reconstruction + depths.pth (EPO's disk path) instead of running VGGT.
The feed-forward init mirrors the disk loaders, so both modes produce
the same refinement.
"""

import argparse
import gc
import os
import sys
import time

import torch

# Make the vendored VGGT + LightGlue submodules importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_THIRD_PARTY = os.path.join(_HERE, "third_party")
for _pkg in ("vggt", "lightglue"):
    _p = os.path.join(_THIRD_PARTY, _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from epo import EPO  # noqa: E402
from wrapper.vggt_wrapper import VGGTWrapper  # noqa: E402


def main():
    """Run VGGT (or load a previous run), refine with EPO, report AUC."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images_path",
        type=str,
        required=True,
        help="Directory containing the input images.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output dir; VGGT writes <output_path>/sparse, "
        "EPO writes <output_path>/sparse_epo.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=150,
        help="Cap on images fed to VGGT (random sample if exceeded). Max on my 4090",
    )
    parser.add_argument(
        "--edges", type=str, default="canny", help="Edge detector for EPO."
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=2000,
        help="Max EPO optimization iterations.",
    )
    parser.add_argument(
        "--early_stop",
        type=str,
        default="pose",
        choices=["none", "pose", "loss"],
        help="EPO early-stopping criterion.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Optional COLMAP GT reconstruction for final AUC eval.",
    )
    parser.add_argument(
        "--vggt_output",
        type=str,
        default=None,
        help="Skip VGGT and load a previous run from this dir (expects the "
        "COLMAP reconstruction + depths.pth). Uses EPO's disk path instead "
        "of the in-memory feed-forward path.",
    )
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument(
        "--vggt_weights",
        type=str,
        default="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        help="Path to VGGT weights (.pt), or a URL (downloaded/cached via "
        "torch.hub). Required by VGGTWrapper; override to use a local checkpoint.",
    )
    args = parser.parse_args()

    t_wall_start = time.perf_counter()  # wall-clock incl. all I/O
    vggt_fwd_time = 0.0  # VGGT model forward (set in the VGGT branch)
    vggt_ff_time = 0.0  # building EPO feed-forward data (decode/resize/crop)
    deferred_writes = []  # disk writes run AFTER the timed inference

    # Shared EPO settings for both the in-memory and disk init paths.
    epo_kwargs = dict(
        detector=args.edges,
        fuse_reduction=True,
        backend="triton",
        use_amp=True,
        max_num_iterations=args.max_iterations,
        single_camera_per_folder=True,  # share one camera across all "<cam_id>/..." images.
        verbose=False,
        log_granular_time=False,
    )

    # ── 1. Reconstruction → EPO init ─────────────────────────────────────
    if args.vggt_output is None:
        # Run VGGT, then init EPO directly from the feed-forward output (no
        # disk round-trip). `_build_ff_data` mirrors EPO's disk loaders, so
        # this is equivalent to reloading the saved reconstruction.
        vggt_out = os.path.join(args.output_path, "sparse_vggt")
        vggt = VGGTWrapper(args.vggt_weights, cuda_id=args.cuda_id)
        # save=False: defer VGGT's disk writes until after the timed inference.
        ff_data, vggt_recon, vggt_depths = vggt.forward(
            args.images_path,
            vggt_out,
            max_images=args.max_images,
            use_ba=False,
            save_depth=True,
            save=False,
        )
        vggt_fwd_time = vggt.last_timings.get("run_vggt", 0.0)
        vggt_ff_time = vggt.last_timings.get("build_ff_data", 0.0)
        del vggt  # free GPU memory before EPO refinement
        gc.collect()
        torch.cuda.empty_cache()
        epo = EPO.from_ff(ff_data, **epo_kwargs)

        def _write_vggt():
            os.makedirs(vggt_out, exist_ok=True)
            vggt_recon.write_text(vggt_out)
            torch.save(vggt_depths, os.path.join(vggt_out, "depths.pth"))

        deferred_writes.append(_write_vggt)
    else:
        # Bypass VGGT: load a previous run's reconstruction + depths from disk.
        vggt_out = args.vggt_output
        print(f"Bypassing VGGT; loading reconstruction + depths from {vggt_out}")
        epo = EPO(
            reconstruction_path=vggt_out,
            images_path=args.images_path,
            depths_path=os.path.join(vggt_out, "depths.pth"),
            **epo_kwargs,
        )

    # ── 2. EPO refinement (timed) ────────────────────────────────────────
    print("\nRunning EPO refinement...")
    epo(early_stop=args.early_stop)
    epo_opt_time = epo.timings.get("total_optimization", 0.0)

    # ── 3. Deferred I/O: write all COLMAP outputs AFTER the timed inference ─
    epo_out = os.path.join(args.output_path, "sparse_epo")
    epo.to_colmap(
        epo_out,
        verbose=False,
        max_points_per_image=100_000 // max(len(epo.images), 1),
        save_points=True,
        final_dbscan_filtering=False,
    )
    for _write in deferred_writes:
        _write()

    epo.print_summary()

    # ── Timing report ─────────────────────────────────────────────────────
    inference_total = vggt_fwd_time + vggt_ff_time + epo_opt_time
    wall_total = time.perf_counter() - t_wall_start
    print(
        f"\nInference time   (VGGT fwd {vggt_fwd_time:.2f}s + ff-build "
        f"{vggt_ff_time:.2f}s + EPO opt {epo_opt_time:.2f}s): {inference_total:.2f} s"
    )
    print(
        f"Total wall-clock (incl. I/O — model load, image load, writes): "
        f"{wall_total:.2f} s"
    )

    # AUC evaluation if GT provided
    if args.gt_path is not None:
        from helpers.benchmark_pose import eval_colmap_model

        thresholds = [1, 3, 5]
        print("AUC@", thresholds)

        AUC_score_max, _, _ = eval_colmap_model(
            vggt_out, args.gt_path, return_df=True, thrs=thresholds
        )
        print(f"{'VGGT AUC:':<16}", [float(round(_, 2)) for _ in AUC_score_max])

        AUC_score_max, _, _ = eval_colmap_model(
            epo_out, args.gt_path, return_df=True, thrs=thresholds
        )
        print(f"{'VGGT + EPO AUC:':<16}", [float(round(_, 2)) for _ in AUC_score_max])


if __name__ == "__main__":
    main()
