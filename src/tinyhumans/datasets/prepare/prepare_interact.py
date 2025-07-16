"""Script to prepare the InterAct dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tinyhumans.datasets.prepare.prepare_base import prepare as prepare_base


def prepare(output_dir: str | Path, quiet: bool = False) -> None:
    """Prepare the InterAct dataset.

    Args:
        output_dir (str): The directory to extract the dataset into.
        quiet (bool): If True, suppresses progress bar and output.

    """
    output_dir = Path(output_dir)
    output_dir = output_dir / "InterAct" if "interact" not in output_dir.as_posix().lower() else output_dir

    files_to_download = [
        {
            "id": "1WMNwEjuUx0IufUW5FvO8krMM_fX_9No5",
            "name": "interact.tar.gz",
            "output_dir": output_dir.parent,
            "gdrive": True,
            "check_exists": output_dir / "chairs",
        },
        {
            "id": "1bGDyTRubPOvBlOCo73B1v1m0Z9D2yv-j",
            "name": "annotations.zip",
            "output_dir": output_dir / "annotation",
            "gdrive": True,
            "check_exists": output_dir / "annotation",
        },
    ]

    prepare_base(output_dir, files_to_download, quiet)

    # make symlinks
    dataset_dir = output_dir.parent
    for dataset in ("behave", "intercap", "omomo", "grab"):
        if not (dataset_dir / dataset).exists():
            os.symlink((dataset_dir / dataset).resolve(), (output_dir / dataset).resolve(), target_is_directory=True)


def main() -> None:
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Prepare the InterAct dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-o", "--output", required=True, help="The destination folder for the dataset.")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress download progress bar and informational messages."
    )

    args = parser.parse_args()

    prepare(args.output, args.quiet)


if __name__ == "__main__":
    main()
