"""End-to-end demo: 3D foundation model reconstruction → EPO refinement.

Runs one of the wrapper/ models (VGGT by default) on a folder of images,
then feeds the resulting COLMAP reconstruction + dense depths to EPO for
pose/depth refinement. Writes the refined reconstruction to
``<output_path>/sparse_epo`` and prints EPO's own timing summary plus the
total end-to-end runtime.

Example:
python demo_epo.py \
    --model vggt \
    --images_path ~/Desktop/datasets/mipnerf360/kitchen/images_150 \
    --output_path optimized_reconstruction/demo \
    --gt_path ~/Desktop/datasets/mipnerf360/kitchen/sparse_150
    --output_path demo/bicycle \
    --gt_path demo_scenes/mipnerf360/bicycle/sparse_150 \
    --densify

Pass ``--model`` to pick a different wrapper/ 3D foundation model (see
``wrapper/__init__.py``'s ``WRAPPERS`` registry for the full list).
Pass ``--model_output <dir>`` to skip the model and reuse a previous run's
reconstruction + depths.pth (EPO's disk path) instead of running it. The
feed-forward init mirrors the disk loaders, so both modes produce the same
refinement.
"""

import argparse
import gc
import os
import time

import torch

from epo import EPO
from wrapper import WRAPPERS, load_wrapper_class


def main():
    """Run the selected model (or load a previous run), refine with EPO, report AUC."""
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
        help="Output dir; --model writes <output_path>/sparse_<model>, "
        "EPO writes <output_path>/sparse_<model>_epo.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vggt",
        choices=sorted(WRAPPERS),
        help="Which wrapper/ 3D foundation model to run.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=150,
        help="Cap on images fed to the model (random sample if exceeded). Max on my 4090",
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
        "--model_output",
        type=str,
        default=None,
        help="Skip --model and load a previous run from this dir (expects "
        "the COLMAP reconstruction + depths.pth). Uses EPO's disk path "
        "instead of the in-memory feed-forward path.",
    )
    parser.add_argument(
        "--densify",
        action="store_true",
        help="After EPO, complete its sparse (edge-only) depth maps with "
        "Any2Full and write a densified model to <output_path>/dense_<model>_epo.",
    )
    parser.add_argument(
        "--densify_batch_size",
        type=int,
        default=4,
        help="Images per Any2Full forward pass when --densify is set.",
    )
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Weights path/URL for --model (downloaded/cached via torch.hub, "
        "or a HF repo id, depending on the wrapper). Defaults to the "
        "WRAPPERS registry entry for --model.",
    )
    args = parser.parse_args()

    t_wall_start = time.perf_counter()  # wall-clock incl. all I/O
    model_fwd_time = 0.0  # model forward (set in the live-run branch)
    model_ff_time = 0.0  # building EPO feed-forward data (decode/resize/crop)
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
    if args.model_output is None:
        # Run the model, then init EPO directly from the feed-forward output
        # (no disk round-trip). `_build_ff_data` mirrors EPO's disk loaders,
        # so this is equivalent to reloading the saved reconstruction.
        model_out = os.path.join(args.output_path, f"sparse_{args.model}")
        wrapper_cls = load_wrapper_class(args.model)
        model_path = args.model_path or WRAPPERS[args.model][2]
        model = wrapper_cls(model_path, cuda_id=args.cuda_id)
        # save=False: defer the model's disk writes until after the timed inference.
        ff_data, model_recon, model_depths = model.forward(
            args.images_path,
            model_out,
            max_images=args.max_images,
            use_ba=False,
            save_depth=True,
            save=False,
        )
        # last_timings has exactly one "run_<model>" key for the forward pass.
        model_fwd_time = next(
            (v for k, v in model.last_timings.items() if k.startswith("run_")), 0.0
        )
        model_ff_time = model.last_timings.get("build_ff_data", 0.0)
        del model  # free GPU memory before EPO refinement
        gc.collect()
        torch.cuda.empty_cache()
        epo = EPO.from_ff(ff_data, **epo_kwargs)

        def _write_model():
            os.makedirs(model_out, exist_ok=True)
            model_recon.write_text(model_out)
            torch.save(model_depths, os.path.join(model_out, "depths.pth"))

        deferred_writes.append(_write_model)
    else:
        # Bypass the model: load a previous run's reconstruction + depths from disk.
        model_out = args.model_output
        print(
            f"Bypassing {args.model}; loading reconstruction + depths from {model_out}"
        )
        epo = EPO(
            reconstruction_path=model_out,
            images_path=args.images_path,
            depths_path=os.path.join(model_out, "depths.pth"),
            **epo_kwargs,
        )

    # ── 2. EPO refinement (timed) ────────────────────────────────────────
    print("\nRunning EPO refinement...")
    epo(early_stop=args.early_stop)
    epo_opt_time = epo.timings.get("total_optimization", 0.0)

    # ── 3. Deferred I/O: write all COLMAP outputs AFTER the timed inference ─
    epo_out = os.path.join(args.output_path, f"sparse_{args.model}_epo")
    epo.to_colmap(
        epo_out,
        verbose=False,
        max_points_per_image=100_000 // max(len(epo.images), 1),
        save_points=True,
        final_dbscan_filtering=args.densify,  # Any2Full's prompt: EPO's edge-only points
        save_depth=args.densify,  # Any2Full's prompt: EPO's edge-only depths
    )
    for _write in deferred_writes:
        _write()

    epo.print_summary()

    # ── Timing report ─────────────────────────────────────────────────────
    inference_total = model_fwd_time + model_ff_time + epo_opt_time
    wall_total = time.perf_counter() - t_wall_start
    print(
        f"\nInference time   ({args.model} fwd {model_fwd_time:.2f}s + ff-build "
        f"{model_ff_time:.2f}s + EPO opt {epo_opt_time:.2f}s): {inference_total:.2f} s"
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
            model_out, args.gt_path, return_df=True, thrs=thresholds
        )
        print(
            f"{args.model + ' AUC:':<16}", [float(round(_, 2)) for _ in AUC_score_max]
        )

        AUC_score_max, _, _ = eval_colmap_model(
            epo_out,
            args.gt_path,
            return_df=True,
            thrs=thresholds,
        )
        print(
            f"{args.model + ' + EPO AUC:':<16}",
            [float(round(_, 2)) for _ in AUC_score_max],
        )

    # ── 4. Optional: densify EPO's edge-only depth maps ───────────────────
    if args.densify:
        # Imported here so the Any2Full repo's generic top-level `model` /
        # `utils` packages only land on sys.path when densification is used.
        from wrapper.any2full_wrapper import Any2FullWrapper, dense_output_path

        del epo  # free GPU memory before loading Any2Full
        gc.collect()
        torch.cuda.empty_cache()

        print("\nDensifying depths with Any2Full...")
        densifier = Any2FullWrapper(cuda_id=args.cuda_id)
        densifier.forward(
            reconstruction_path=epo_out,
            images_path=args.images_path,
            batch_size=args.densify_batch_size,
        )
        print(
            f"Densification time: {densifier.last_timings['run_any2full']:.2f} s "
            f"-> {dense_output_path(epo_out)}"
        )


if __name__ == "__main__":
    main()
