# Since pycolmap BA is capped to 100 iterations, I run BA from CLI on top of it without limit of iterations so the model can converge.
# I use the same options as in pycolmap BA and add only BA running time computed by COLMAP.

import re
import os
import glob
import subprocess


ds = "mipnerf360"  # terrasky3D, mipnerf360, scannetpp
paths = sorted(glob.glob(f"benchmarks/vggt_ba/{ds}/*/sparse"))

# paths = [p for p in paths if "treehill" not in p]

for path in paths:
    # if not ("graz_church" in path or "graz_university" in path):
    #     continue  # testing only these two for now

    print(f"Processing path: {path}")

    # read timings.txt for reference
    with open(f"{path}/timings.txt", "r") as f:
        raw_output = f.read()

    # Convert to dictionary
    # We split by newline, then split by ':' and take the first word of the value (the number)
    stats_dict = {
        line.split(":")[0].strip(): float(line.split(":")[1].strip().split(" ")[0])
        for line in raw_output.strip().split("\n")
    }

    # Run Bundle Adjustment to convergence, pycolmap is bounded to 100 iterations
    cmd = [
        "colmap",
        "bundle_adjuster",
        "--input_path",
        f"{path}",
        "--output_path",
        f"{path}",
        "--BundleAdjustment.max_num_iterations",
        "500",
        "--BundleAdjustment.refine_principal_point",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Now your output is in a variable
    ba_output = result.stderr
    # 1. Extract Time (specifically the one followed by [s])
    time_match = re.search(r"Time\s+:\s+([\d.]+)\s+\[s\]", ba_output)
    ba_time = float(time_match.group(1)) if time_match else None

    # 2. Extract Termination status
    term_match = re.search(r"Termination\s+:\s+(\w+)", ba_output)
    termination = term_match.group(1) if term_match else None

    # detect "iterations"
    iter_match = re.search(r"Iterations\s+:\s+(\d+)", ba_output)
    ba_iterations = int(iter_match.group(1)) if iter_match else None

    if ba_iterations == 1:  # means it was not needed so we skip
        print("Bundle Adjustment already converged.")
        continue

    print(f"Extracted Time: {ba_time} seconds")
    print(f"Termination Status: {termination}")
    print(f"Bundle Adjustment Output:\n{ba_output}")

    stats_dict["convergence"] = bool(termination == "Convergence")
    stats_dict["reconstruction_with_ba"] += ba_time
    stats_dict["total"] += ba_time

    stats_dict = {k: round(v, 2) for k, v in stats_dict.items()}

    with open(f"{path}/timings.txt", "w") as f:
        for k, v in stats_dict.items():
            if k != "total":
                f.write(f"{k}: {v}\n")
        f.write(f"total: {stats_dict['total']}\n")

    # We remove standard .txt model files if they exist to force usage of new .bin files.
    model_txt_files = ["cameras.txt", "images.txt", "points3D.txt"]
    for filename in model_txt_files:
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            print(f"Removing old model file to prevent conflicts: {filename}")
            os.remove(file_path)
