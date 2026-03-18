"""Package a submission directory into a Codabench-ready ZIP file.

Usage
-----
    python package_submission.py <submission_dir> [--out <zip_path>]

Examples
--------
    # Package submission 1 (random agent — no weights needed)
    python package_submission.py submissions/submission1_random

    # Package submission 3 (Q-learning — requires qtable.npy to be present)
    python package_submission.py submissions/submission3_qlearning

    # Package submission 4 with a custom output name
    python package_submission.py submissions/submission4_ddqn --out my_ddqn_v2.zip

What gets included
------------------
Every file inside the submission directory *except* hidden files and
__pycache__ directories is added to the root of the zip (no nested folder).
This matches Codabench's expected layout where agent.py is at the top level.

After packaging
---------------
Upload the resulting .zip on Codabench under:
    My Submissions → Submit → choose the .zip file
"""

import argparse
import os
import zipfile


def package(submission_dir: str, out_zip: str) -> None:
    submission_dir = os.path.abspath(submission_dir)

    if not os.path.isdir(submission_dir):
        raise NotADirectoryError(f"Not a directory: {submission_dir}")

    files_to_include = []
    for root, dirs, files in os.walk(submission_dir):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for fname in files:
            if fname.startswith("."):
                continue
            full_path = os.path.join(root, fname)
            # Archive path: relative to the submission directory root
            arcname = os.path.relpath(full_path, submission_dir)
            files_to_include.append((full_path, arcname))

    if not files_to_include:
        raise RuntimeError(f"No files found in {submission_dir}")

    # Verify agent.py exists
    has_agent = any(arc == "agent.py" for _, arc in files_to_include)
    if not has_agent:
        raise RuntimeError(
            f"agent.py not found in {submission_dir}. "
            "Each submission must contain an agent.py that defines policy()."
        )

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for full_path, arcname in sorted(files_to_include, key=lambda x: x[1]):
            zf.write(full_path, arcname)
            print(f"  + {arcname}")

    total_kb = os.path.getsize(out_zip) / 1024
    print(f"\nCreated {out_zip} ({total_kb:.1f} KB)  — {len(files_to_include)} file(s)")
    print("Ready to upload to Codabench ✓")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package a submission directory into a Codabench-ready ZIP."
    )
    parser.add_argument(
        "submission_dir",
        help="path to the submission folder (e.g. submissions/submission3_qlearning)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output zip path (default: <submission_dir_name>.zip in current directory)",
    )
    args = parser.parse_args()

    out_zip = args.out or (os.path.basename(args.submission_dir.rstrip("/")) + ".zip")
    package(args.submission_dir, out_zip)


if __name__ == "__main__":
    main()
