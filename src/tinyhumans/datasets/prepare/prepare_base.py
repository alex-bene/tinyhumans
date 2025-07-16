"""Base script for dataset preparation."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import gdown
import requests
from tqdm import tqdm

from tinyhumans.tools import get_logger

logger = get_logger(__name__, "info")


def safe_zip_extract_all(zip_ref: zipfile.ZipFile, output_dir: str | Path) -> None:
    """Extract all files from a zip archive, even if they are nested in a folder."""
    output_dir = Path(output_dir)
    safe_dest_dir = output_dir.resolve()
    for member in zip_ref.infolist():
        member_path = safe_dest_dir / member.filename
        if not member_path.resolve().as_posix().startswith(safe_dest_dir.as_posix()):
            logger.warning("Skipping potentially unsafe member: %s", member.filename)
            continue
        zip_ref.extract(member, path=output_dir)


def safe_tar_extract_all(tar_ref: tarfile.TarFile, output_dir: str | Path) -> None:
    """Extract all files from a zip archive, even if they are nested in a folder."""
    output_dir = Path(output_dir)
    safe_dest_dir = output_dir.resolve()
    for member in tar_ref.getmembers():
        member_path = safe_dest_dir / member.name
        if not member_path.resolve().as_posix().startswith(safe_dest_dir.as_posix()):
            logger.warning("Skipping potentially unsafe member: %s", member.name)
            continue
        tar_ref.extract(member, path=output_dir)


def download_from_url(
    url: str, post_data: dict | None = None, download_path: str | Path = ".", quiet: bool = False
) -> None:
    """Download a file from a generic URL with a TQDM progress bar."""
    download_path = Path(download_path)

    # --- Resuming Logic ---
    headers = {}
    existing_size = 0
    if download_path.exists():
        existing_size = download_path.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"  # Request to resume
        logger.info("Resuming download of %s from byte %s", url, existing_size)
    # --- End Resuming Logic ---

    req_type = requests.post if post_data is not None else requests.get
    response = req_type(url, data=post_data, stream=True, headers=headers, timeout=10)
    response.raise_for_status()

    # Check if the server respected the Range request (status code 206)
    is_resumed = response.status_code == 206
    if is_resumed:
        content_range = response.headers["Content-Range"]
        total_size = int(content_range.split("/")[1])  # Extract total size from Content-Range
    else:
        existing_size = 0
        headers = {}
        response = req_type(url, data=post_data, headers=headers, stream=True, timeout=10)  # redownload
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

    with (
        tqdm(total=total_size, unit="iB", unit_scale=True, initial=existing_size, disable=quiet) as progress_bar,
        download_path.open("ab" if is_resumed else "wb") as f,  # Append or write
    ):
        for chunk in response.iter_content(chunk_size=1024):
            progress_bar.update(len(chunk))
            f.write(chunk)

    # check final size is correct
    if download_path.exists() and download_path.stat().st_size != total_size:
        msg = "Download failed: Mismatch in file size."
        raise OSError(msg)


def prepare(output_dir: str | Path, files_to_download: list[dict], quiet: bool = False) -> None:  # noqa: PLR0912
    """Download and extracts files for various dataset.

    Args:
        output_dir (str): The directory to extract the dataset into.
        files_to_download (list[dict]): A list of dictionaries containing information about the files to download.
        quiet (bool): If True, suppresses progress bar and output.

    """
    output_dir = Path(output_dir)

    for file_info in files_to_download:
        file_id = file_info["id"]
        file_name = file_info["name"]
        is_gdrive = file_info["gdrive"]
        post_data = file_info.get("post_data", None)
        check_exists: Path = file_info["check_exists"]
        output_dir_cur: Path = file_info["output_dir"]
        download_path: Path = output_dir_cur / file_name

        # if folder exists skip
        if check_exists.exists():
            logger.info("Skipping %s as it has already been downloaded & extracted.", file_name)
            continue

        output_dir_cur.mkdir(exist_ok=True, parents=True)

        if not quiet:
            logger.info("Processing %s...", file_name)
            logger.info("Downloading...")

        # Download the file using its ID
        if is_gdrive:
            gdown.download(id=file_id, output=download_path.as_posix(), quiet=quiet)
        else:
            download_from_url(file_id, post_data=post_data, download_path=download_path, quiet=quiet)

        if not download_path.exists() or download_path.stat().st_size == 0:
            msg = f"Download failed. {download_path} is missing or empty."
            raise OSError(msg)

        if not quiet:
            logger.info("Successfully downloaded to %s", download_path)
            logger.info("Extracting files...")

        # Extract the archive
        if file_name.endswith((".tar.gz", ".tar")):
            if not tarfile.is_tarfile(download_path):
                msg = f"The downloaded file at {download_path} is not a valid tar archive."
                raise ValueError(msg)
            with tarfile.open(download_path, "r") as tar:
                safe_tar_extract_all(tar, output_dir_cur)
        elif file_name.endswith(".zip"):
            if not zipfile.is_zipfile(download_path):
                msg = f"The downloaded file at {download_path} is not a valid zip archive."
                raise ValueError(msg)
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                safe_zip_extract_all(zip_ref, output_dir_cur)

        # Clean up the downloaded archive file
        if not quiet:
            logger.info("Successfully extracted dataset to %s", output_dir_cur)
            logger.info("Cleaning up %s...", file_name)
        download_path.unlink(missing_ok=True)

    if not quiet:
        logger.info("Process finished successfully.")
