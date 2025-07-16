"""Archive extraction tools for TinyHumans."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tinyhumans.tools.logger import get_logger

if TYPE_CHECKING:
    import tarfile
    import zipfile

logger = get_logger(__name__)


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
