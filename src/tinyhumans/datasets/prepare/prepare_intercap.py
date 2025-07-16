"""Script to prepare the InterCap dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tinyhumans.datasets.prepare.prepare_base import prepare as prepare_base


def prepare(output_dir: str | Path, username: str, password: str, quiet: bool = False) -> None:
    """Prepare the InterCap dataset.

    Args:
        output_dir (str): The directory to extract the dataset into.
        username (str): Username for InterCap.
        password (str): Password for InterCap.
        quiet (bool): If True, suppresses progress bar and output.

    """
    output_dir = Path(output_dir)
    output_dir = output_dir / "intercap" if "intercap" not in output_dir.as_posix().lower() else output_dir
    output_dir = output_dir / "raw"

    # --- Base URL for Downloads ---
    base_url = "https://download.is.tue.mpg.de/download.php"
    domain = "intercap"

    # --- Download files ---
    files_to_download = [
        {
            "id": f"{base_url}?domain={domain}&resume=1&sfile={f'Res2_Individuals//{i:02d}.zip'}",
            "name": f"{i:02d}.zip",
            "output_dir": output_dir,
            "gdrive": False,
            "post_data": {"username": username, "password": password},
            "check_exists": output_dir / f"{i:02d}",
        }
        for i in range(1, 11)
    ]

    prepare_base(output_dir, files_to_download, quiet)


def main() -> None:
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Prepare the InterCap dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--username", help="Username.")
    parser.add_argument("--password", help="Password.")
    parser.add_argument("-o", "--output", required=True, help="The destination folder for the dataset.")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress download progress bar and informational messages."
    )

    args = parser.parse_args()

    # --- Get Credentials (prioritize args, then env vars) ---
    username = args.username or os.environ.get("INTERCAP_USERNAME") or os.environ.get("MPI_USERNAME")
    password = args.password or os.environ.get("INTERCAP_PASSWORD") or os.environ.get("MPI_PASSWORD")

    prepare(args.output, username, password, args.quiet)


if __name__ == "__main__":
    main()
