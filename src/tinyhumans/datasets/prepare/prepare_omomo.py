"""Script to prepare the OMOMO dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tinyhumans.datasets.prepare.prepare_base import prepare as prepare_base


def prepare(output_dir: str | Path, quiet: bool = False) -> None:
    """Prepare the OMOMO dataset.

    Args:
        output_dir (str): The directory to extract the dataset into.
        quiet (bool): If True, suppresses progress bar and output.

    """
    output_dir = Path(output_dir)
    output_dir = output_dir / "omomo" if "omomo" not in output_dir.as_posix().lower() else output_dir

    files_to_download = [
        {
            "id": "1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm",
            "name": "data.tar.gz",
            "output_dir": output_dir,
            "gdrive": True,
            "check_exists": output_dir / "raw",
        },
        {
            "id": "https://github.com/lijiaman/omomo_release/raw/refs/heads/main/omomo_text_anno.zip",
            "name": "omomo_text_anno.zip",
            "output_dir": output_dir / "raw",
            "gdrive": False,
            "check_exists": output_dir / "raw" / "omomo_text_anno_json_data",
        },
    ]

    prepare_base(output_dir, files_to_download, quiet)
    if (output_dir / "data").exists():
        shutil.move(output_dir / "data", output_dir / "raw")


def main() -> None:
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Prepare the OMOMO dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-o", "--output", required=True, help="The destination folder for the dataset.")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress download progress bar and informational messages."
    )

    args = parser.parse_args()

    prepare(args.output, args.quiet)


if __name__ == "__main__":
    main()
