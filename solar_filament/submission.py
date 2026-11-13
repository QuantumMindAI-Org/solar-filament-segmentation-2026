"""The submission CSV in the organisers' format.

Two columns, filament_id and segmentation_rle. Each row is one predicted filament. The id is the
image id, an underscore and a 4-digit running number. The RLE is COCO compressed counts for a
2048 x 2048 mask. An image with no kept component writes no row. Lines end with a line feed.
"""
import csv
import hashlib
import os
import re
from pathlib import Path

COLUMNS = ("filament_id", "segmentation_rle")
FILAMENT_ID = re.compile(r"([0-9]{14}[A-Za-z]{2})_([A-Za-z0-9]+)\Z")


def check_rows(rows):
    """Refuse a row that the organisers' format would not accept."""
    for filament_id, rle in rows:
        if FILAMENT_ID.fullmatch(filament_id) is None:
            raise ValueError("FILAMENT_ID")
        if not (isinstance(rle, str) and rle.isascii() and rle):
            raise ValueError("RLE")


def write_csv(path, rows):
    """Write rows (an iterable of (filament_id, rle)) to path, atomically; return the file's sha256."""
    rows = list(rows)
    check_rows(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:      # LF on every platform
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    os.replace(temporary, target)
    return hashlib.sha256(target.read_bytes()).hexdigest()
