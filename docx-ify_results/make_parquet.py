from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from docx_helper import (
    extract_run_template,
    build_field_chunks,
    parse_run_dir,
)


ROOT = (
    Path(os.environ["CMB_ML_DATA"]) / 
    "Datasets/I_32_Inf_CMB_only/train_otf"
)

DIRNAME_SPEC = (
    "pred_${spectrum_type}-lr_${learning_rate}-lbl_${prep_label}-ftr_${prep_feature}-fcn_str_${fcn_string}-act_${activation}-lmax_${lmax}/"
)

LOSS_REL_PATH = Path("F_Model/loss_record.csv")

OUT_DIR = ROOT / "_cache"
OUT_PATH = OUT_DIR / "loss_records.parquet"

RUN_TEMPLATE = extract_run_template(DIRNAME_SPEC)
FIELD_CHUNKS = build_field_chunks(RUN_TEMPLATE)
PARAM_KEYS = [chunk.combined_name for chunk in FIELD_CHUNKS]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])

    dfs = []

    print(f"Found {len(run_dirs)} directories.")

    for i, run_dir in enumerate(run_dirs, start=1):
        rec = parse_run_dir(
            run_dir=run_dir,
            fig_subdir="",
            fig_files={},
            field_chunks=FIELD_CHUNKS,
        )

        if rec is None:
            print(f"[skip] could not parse: {run_dir.name}")
            continue

        loss_path = run_dir / LOSS_REL_PATH

        if not loss_path.exists():
            print(f"[skip] missing: {loss_path}")
            continue

        try:
            df = pd.read_csv(loss_path)
        except Exception as e:
            print(f"[skip] failed reading {loss_path}")
            print(f"       {e}")
            continue

        # Ensure Epoch exists.
        if "Epoch" not in df.columns:
            df.insert(0, "Epoch", np.arange(1, len(df) + 1))

        # Add identifying info.
        df.insert(0, "run_stem", rec.stem)

        for key in PARAM_KEYS:
            df[key] = str(rec.params.get(key, ""))

        dfs.append(df)

        if i % 100 == 0:
            print(f"Processed {i}/{len(run_dirs)}")

    if not dfs:
        raise RuntimeError("No loss records loaded.")

    full_df = pd.concat(dfs, ignore_index=True)

    print("\nFinal dataframe:")
    print(full_df.shape)

    print("\nColumns:")
    for c in full_df.columns:
        print(f"  {c}")

    print(f"\nWriting parquet:")
    print(f"  {OUT_PATH}")

    full_df.to_parquet(OUT_PATH, index=False)

    size_mb = OUT_PATH.stat().st_size / (1024 ** 2)

    print("\nDone.")
    print(f"Parquet size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
