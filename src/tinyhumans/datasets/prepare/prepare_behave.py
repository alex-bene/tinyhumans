"""Script to prepare the BEHAVE dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinyhumans.datasets.prepare.prepare_base import prepare as prepare_base


def prepare(output_dir: str | Path, quiet: bool = False) -> None:
    """Prepare the BEHAVE dataset.

    Args:
        output_dir (str): The directory to extract the dataset into.
        quiet (bool): If True, suppresses progress bar and output.

    """
    output_dir = Path(output_dir)
    output_dir = output_dir / "behave" if "behave" not in output_dir.as_posix().lower() else output_dir

    files_to_download = [
        {
            "id": "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-30fps-params-v1.tar",
            "name": "behave-30fps-params-v1.tar",
            "output_dir": output_dir / "sequences",
            "gdrive": False,
            "check_exists": output_dir / "sequences",
        },
        {
            "id": "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/objects.zip",
            "name": "objects.zip",
            "output_dir": output_dir,
            "gdrive": False,
            "check_exists": output_dir / "objects",
        },
    ]

    prepare_base(output_dir, files_to_download, quiet)


def main() -> None:
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Prepare the BEHAVE dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-o", "--output", required=True, help="The destination folder for the dataset.")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress download progress bar and informational messages."
    )

    args = parser.parse_args()

    prepare(args.output, args.quiet)


if __name__ == "__main__":
    main()
