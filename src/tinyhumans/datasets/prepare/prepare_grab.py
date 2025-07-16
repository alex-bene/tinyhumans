"""Script to prepare the GRAB dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tinyhumans.datasets.prepare.prepare_base import prepare as prepare_base


def prepare(output_dir: str | Path, username: str, password: str, quiet: bool = False) -> None:
    """Prepare the GRAB dataset.

    Args:
        output_dir (str): The directory to extract the dataset into.
        username (str): Username for InterCap.
        password (str): Password for InterCap.
        quiet (bool): If True, suppresses progress bar and output.

    """
    output_dir = Path(output_dir)
    output_dir = output_dir / "grab" if "grab" not in output_dir.as_posix().lower() else output_dir
    output_dir = output_dir / "raw"

    # --- Base URL for Downloads ---
    base_url = "https://download.is.tue.mpg.de/download.php"
    domain = "grab"

    # --- Download files ---
    files_to_download = [
        {
            "id": f"{base_url}?domain={domain}&resume=1&sfile={f'grab__s{i}.zip'}",
            "name": f"grab__s{i}.zip",
            "output_dir": output_dir / "grab",
            "gdrive": False,
            "post_data": {"username": username, "password": password},
            "check_exists": output_dir / "grab" / f"s{i}",
        }
        for i in range(1, 11)
    ] + [
        {
            "id": f"{base_url}?domain={domain}&resume=1&sfile=tools__subject_meshes__female.zip",
            "name": "tools__subject_meshes__female.zip",
            "output_dir": output_dir / "tools" / "subject_meshes",
            "gdrive": False,
            "post_data": {"username": username, "password": password},
            "check_exists": output_dir / "tools" / "subject_meshes" / "female",
        },
        {
            "id": f"{base_url}?domain={domain}&resume=1&sfile=tools__subject_meshes__male.zip",
            "name": "tools__subject_meshes__male.zip",
            "output_dir": output_dir / "tools" / "subject_meshes",
            "gdrive": False,
            "post_data": {"username": username, "password": password},
            "check_exists": output_dir / "tools" / "subject_meshes" / "male",
        },
        {
            "id": f"{base_url}?domain={domain}&resume=1&sfile=tools__object_settings.zip",
            "name": "tools__object_settings.zip",
            "output_dir": output_dir / "tools",
            "gdrive": False,
            "post_data": {"username": username, "password": password},
            "check_exists": output_dir / "tools" / "object_settings",
        },
        {
            "id": f"{base_url}?domain={domain}&resume=1&sfile=tools__subject_settings.zip",
            "name": "tools__subject_settings.zip",
            "output_dir": output_dir / "tools",
            "gdrive": False,
            "post_data": {"username": username, "password": password},
            "check_exists": output_dir / "tools" / "subject_settings",
        },
        {
            "id": f"{base_url}?domain={domain}&resume=1&sfile=tools__smplx_correspondence.zip",
            "name": "tools__smplx_correspondence.zip",
            "output_dir": output_dir / "tools",
            "gdrive": False,
            "post_data": {"username": username, "password": password},
            "check_exists": output_dir / "tools" / "smplx_correspondence",
        },
        {
            "id": f"{base_url}?domain={domain}&resume=1&sfile=contactdb_scaled_stl_files_public.zip",
            "name": "contactdb_scaled_stl_files_public.zip",
            "output_dir": output_dir / "tools" / "object_meshes",
            "gdrive": False,
            "post_data": {"username": username, "password": password},
            "check_exists": output_dir / "tools" / "object_meshes" / "airplane.stl",
        },
        {
            "id": f"{base_url}?domain={domain}&resume=1&sfile=tools__object_meshes__contact_meshes.zip",
            "name": "tools__object_meshes__contact_meshes.zip",
            "output_dir": output_dir / "tools" / "object_meshes",
            "gdrive": False,
            "post_data": {"username": username, "password": password},
            "check_exists": output_dir / "tools" / "object_meshes" / "contact_meshes",
        },
    ]

    prepare_base(output_dir, files_to_download, quiet)


def main() -> None:
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Prepare the GRAB dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--username", help="Username.")
    parser.add_argument("--password", help="Password.")
    parser.add_argument("-o", "--output", required=True, help="The destination folder for the dataset.")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress download progress bar and informational messages."
    )

    args = parser.parse_args()

    # --- Get Credentials (prioritize args, then env vars) ---
    username = args.username or os.environ.get("GRAB_USERNAME") or os.environ.get("MPI_USERNAME")
    password = args.password or os.environ.get("GRAB_PASSWORD") or os.environ.get("MPI_PASSWORD")

    prepare(args.output, username, password, args.quiet)


if __name__ == "__main__":
    main()
