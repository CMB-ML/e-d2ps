from __future__ import annotations

import os
import tarfile
from pathlib import Path

# Same ROOT as the report script: the directory holding the per-run subdirectories
# whose names encode the sweep parameters.
ROOT = (
    Path(os.environ["CMB_ML_DATA"]) /
    "Datasets/I_32_Inf_CMB_only/otf_cmb2ps_fcn_scales"
)

# Extensions to archive (lowercase, leading dot). Matched case-insensitively.
IMAGE_EXTS = {".png", ".pdf"}

# Where to write the archive. Placed beside ROOT so it isn't swept up in its own glob.
OUT_PATH = ROOT.parent / f"{ROOT.name}_figures.tar.gz"

# Top-level folder inside the archive, so extraction stays tidy.
ARCHIVE_PREFIX = ROOT.name

# Set True to list what would be archived without writing anything.
DRY_RUN = False


def find_image_files(root: Path, exts: set[str]) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def main() -> None:
    if not ROOT.is_dir():
        raise RuntimeError(f"ROOT is not a directory: {ROOT}")

    files = find_image_files(ROOT, IMAGE_EXTS)
    if not files:
        raise RuntimeError(f"No {sorted(IMAGE_EXTS)} files found under {ROOT}")

    total_bytes = sum(p.stat().st_size for p in files)
    print(f"Found {len(files)} files ({total_bytes / 1e6:.1f} MB) under:\n  {ROOT}")

    if DRY_RUN:
        for p in files:
            arcname = Path(ARCHIVE_PREFIX) / p.relative_to(ROOT)
            print(f"  [dry-run] {arcname}")
        print(f"\n[dry-run] Would write: {OUT_PATH}")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT_PATH, "w:gz") as tar:
        for p in files:
            # arcname keeps <run_dir>/.../<file> (prefixed by ARCHIVE_PREFIX), so
            # per-run figures with identical filenames stay separate and the run
            # parameters are recoverable from the directory names.
            arcname = Path(ARCHIVE_PREFIX) / p.relative_to(ROOT)
            tar.add(p, arcname=str(arcname))

    print(f"\nWrote {len(files)} files to:\n  {OUT_PATH}")
    print(f"Archive size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()