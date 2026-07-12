"""Batch-run a 3D foundation model over benchmark datasets.

The 3DFM is selected via ``--3dfm`` (e.g. ``vggt``, ``vggt_omega``); dataset
roots live in a private JSON file next to this script (``paths.json``, see
``paths_example.json`` for the expected layout).

Examples:
    python wrapper/run_for_dataset.py --3dfm vggt --dataset mipnerf360_local
    python wrapper/run_for_dataset.py --3dfm vggt --dataset eth3d_local tt_local --ba
"""

import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wrapper import WRAPPERS, load_wrapper_class  # noqa: E402


def list_scenes(base_path: str, exclude: list[str]) -> list[str]:
    """List scene subfolders of ``base_path``, skipping excluded/hidden ones."""
    return [
        s
        for s in sorted(os.listdir(base_path))
        if s not in exclude
        and not s.startswith(".")  # skip e.g. .vscode
        and os.path.isdir(os.path.join(base_path, s))  # skip files e.g. gen_data.py
    ]


def run_dataset(args, dataset: str, cfg: dict, wrapper_cls, model_path: str):
    """Run the selected 3DFM on every scene of one dataset."""
    _ba = "_ba" if args.ba else ""
    base_path = cfg["base_path"]
    images_path = cfg["images_path"]  # I might have files in that path
    output_folder = cfg["output_path"]

    model = None
    if not args.ba:
        model = wrapper_cls(model_path, cuda_id=args.cuda_id)

    for scene in list_scenes(base_path, cfg.get("exclude", [])):
        input_path = f"{base_path}/{scene}/{images_path}"
        output_path = (
            f"{output_folder}/{args.model}{_ba}/{dataset.split('_')[0]}/{scene}/sparse"
        )

        if not os.path.exists(input_path):
            print(f"Input path {input_path} does not exist, skipping...")
            continue

        if os.path.exists(output_path + "/cameras.txt"):
            print(f"Output path {output_path} already exists, skipping...")
            continue

        num_images = len(
            glob.glob(f"{input_path}/*.*")
            + glob.glob(f"{input_path}/*/*.*", recursive=True)
        )
        if num_images == 0:
            print(f"No images found in {input_path}, skipping...")
            continue

        print(f"Processing {dataset} - {scene}...")

        if args.ba:
            # Fresh instance per scene so oom_safe can free the model.
            model = wrapper_cls(model_path, cuda_id=args.cuda_id, oom_safe=True)

        _ = model.forward(
            input_path,
            output_path,
            max_images=-1,
            use_ba=args.ba,
            query_frame_num=10,  # 10 (or 8)
            max_query_pts=2048,  # // 2,
            fine_tracking=False,  # True,
            shared_camera=True,  # per folder, in my datasets usually true
        )


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--3dfm",
        dest="model",
        default="vggt",
        choices=sorted(WRAPPERS),
        help="Which 3D foundation model wrapper to run.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="Dataset key(s) from the paths JSON, e.g. mipnerf360_local.",
    )
    parser.add_argument(
        "--paths",
        default=os.path.join(_HERE, "paths.json"),
        help="JSON with dataset roots (copy paths_example.json to create it).",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Weights path/URL; defaults to the registry entry for --3dfm.",
    )
    parser.add_argument(
        "--ba", action="store_true", help="Run bundle adjustment."
    )  # works only of VGGT
    parser.add_argument("--cuda_id", type=int, default=0)
    args = parser.parse_args()

    with open(args.paths) as f:
        paths = json.load(f)

    unknown = [d for d in args.dataset if d not in paths]
    if unknown:
        parser.error(f"Unknown dataset(s) {unknown}; available: {sorted(paths)}")

    wrapper_cls = load_wrapper_class(args.model)
    model_path = args.model_path or WRAPPERS[args.model][2]

    for dataset in args.dataset:
        run_dataset(args, dataset, paths[dataset], wrapper_cls, model_path)


if __name__ == "__main__":
    main()
