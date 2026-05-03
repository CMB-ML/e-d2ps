from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from docx import Document

from docx_helper import (
    extract_run_template,
    build_field_chunks,
    RunRecord,
    parse_run_dir,
    find_varying_keys,
)


ROOT = (
    Path(os.environ["CMB_ML_DATA"])
    / "Datasets/I_128_20850/cmb2ps_SimpleEncDec_train_reruns"
)

DIRNAME_SPEC = (
    "Nsims-${n_sims}_lr-${learning_rate}_${optimizer}_wd-${adamw_wd}_sps-${scale_ps}_spx-${model.scale_px}_pe-${use_pos_enc_lut}_en-${norm_kind}_${activation}_r1-${down_reductions}_r2-${spatial_reduction}_${model_num}/"
)

LOSS_REL_PATH = Path("F_Model/loss_record.csv")
LOSS_COLUMN = "Validation Loss"

LOSS_MODE = "min_smoothed"  # "min_raw" or "min_smoothed"
SMOOTH_SIGMA = 5

COLUMN_KEYS = [
    "down_reductions",
]

ROW_KEYS = [
    "adamw_wd",
    "learning_rate",
]

FIELD_VALUE_ORDER = {
    "down_reductions": ["max", "Xm", "XXm", "mXX", "avg"],
}

FIG_SUBDIR = "Analysis_Figures"
FIG_FILES = {}

RUN_TEMPLATE = extract_run_template(DIRNAME_SPEC)
FIELD_CHUNKS = build_field_chunks(RUN_TEMPLATE)
PARAM_KEYS = [chunk.combined_name for chunk in FIELD_CHUNKS]


def parse_number_string(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def is_exp_notation(s: str) -> bool:
    return "e" in str(s).lower()


def all_values_are_number_strings(values: list[str]) -> bool:
    return all(parse_number_string(v) is not None for v in values)


def format_exp_no_padded_zero(x: float) -> str:
    s = f"{x:.15g}"

    if "e" not in s.lower():
        s = f"{x:.15e}"

    mant, exp = s.lower().split("e")
    mant = mant.rstrip("0").rstrip(".")
    exp_i = int(exp)

    return f"{mant}e{exp_i}"


def should_force_exp(values: list[str]) -> bool:
    return (
        all_values_are_number_strings(values)
        and any(is_exp_notation(v) for v in values)
    )


def build_field_value_formatter(
    records: list[RunRecord],
    keys: list[str],
) -> dict[str, dict[str, str]]:
    formatters = {}

    for key in keys:
        raw_values = [str(rec.params.get(key, "")) for rec in records]
        unique_values = sorted(set(raw_values), key=str)

        if should_force_exp(unique_values):
            formatters[key] = {
                v: format_exp_no_padded_zero(float(v))
                for v in unique_values
            }
        else:
            formatters[key] = {v: v for v in unique_values}

    return formatters


def field_value_sort_key(key: str, value: str):
    value = "" if value is None else str(value)

    explicit_order = FIELD_VALUE_ORDER.get(key)
    if explicit_order is not None:
        if value in explicit_order:
            return (0, explicit_order.index(value))
        return (1, value)

    numeric_value = parse_number_string(value)
    if numeric_value is not None:
        return (0, numeric_value)

    return (1, value)


def format_field_value(
    key: str,
    value: str,
    value_formatters: dict[str, dict[str, str]],
) -> str:
    value = str(value)
    return value_formatters.get(key, {}).get(value, value)


def get_key_tuple(rec: RunRecord, keys: list[str]) -> tuple[str, ...]:
    return tuple(str(rec.params.get(k, "")) for k in keys)


def axis_tuple_sort_key(keys: list[str], t: tuple[str, ...]):
    return tuple(
        field_value_sort_key(key, value)
        for key, value in zip(keys, t)
    )


def get_unique_axis_tuples(
    records: list[RunRecord],
    keys: list[str],
) -> list[tuple[str, ...]]:
    return sorted(
        {get_key_tuple(rec, keys) for rec in records},
        key=lambda t: axis_tuple_sort_key(keys, t),
    )


def value_repeats_every(
    col_tuples: list[tuple[str, ...]],
    level: int,
) -> list[tuple[int, int, str]]:
    if not col_tuples:
        return []

    spans = []
    start = 0
    current = col_tuples[0][level]

    for i in range(1, len(col_tuples)):
        val = col_tuples[i][level]
        if val != current:
            spans.append((start, i, current))
            start = i
            current = val

    spans.append((start, len(col_tuples), current))
    return spans


def _find_dupes(keys: list[str]) -> list[str]:
    seen = set()
    dupes = []
    for k in keys:
        if k in seen and k not in dupes:
            dupes.append(k)
        seen.add(k)
    return dupes


def validate_axis_keys(
    varying_keys: list[str],
    row_keys: list[str],
    col_keys: list[str],
) -> None:
    row_dupes = _find_dupes(row_keys)
    if row_dupes:
        raise ValueError(f"Duplicate keys in ROW_KEYS: {row_dupes}")

    col_dupes = _find_dupes(col_keys)
    if col_dupes:
        raise ValueError(f"Duplicate keys in COLUMN_KEYS: {col_dupes}")

    overlap = [k for k in row_keys if k in set(col_keys)]
    if overlap:
        raise ValueError(f"Keys appear in both ROW_KEYS and COLUMN_KEYS: {overlap}")

    varying_set = set(varying_keys)

    missing_rows = [k for k in row_keys if k not in varying_set]
    if missing_rows:
        print(f"[warn] ROW_KEYS contains keys that are not varying: {missing_rows}")

    missing_cols = [k for k in col_keys if k not in varying_set]
    if missing_cols:
        print(f"[warn] COLUMN_KEYS contains keys that are not varying: {missing_cols}")

    unassigned = [
        k for k in varying_keys
        if k not in set(row_keys) and k not in set(col_keys)
    ]
    if unassigned:
        print(
            "[warn] varying keys not assigned to rows/columns. "
            "They will be ignored in the table layout: "
            f"{unassigned}"
        )


def format_loss_value(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "--"

    s = f"{x:.2e}"
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return s


def read_best_validation_loss(run_dir: Path) -> float | None:
    path = run_dir / LOSS_REL_PATH

    if not path.exists():
        print(f"[warn] missing loss file: {path}")
        return None

    df = pd.read_csv(path)

    if LOSS_COLUMN not in df.columns:
        print(f"[warn] missing column {LOSS_COLUMN!r} in {path}")
        return None

    vals = pd.to_numeric(df[LOSS_COLUMN], errors="coerce")
    vals = vals.dropna().to_numpy(dtype=float)

    if len(vals) == 0:
        print(f"[warn] no finite validation losses in {path}")
        return None

    if LOSS_MODE == "min_raw":
        return float(np.min(vals))

    if LOSS_MODE == "min_smoothed":
        if len(vals) == 1:
            return float(vals[0])

        log_vals = np.log(vals)
        smoothed_log = gaussian_filter1d(log_vals, sigma=SMOOTH_SIGMA)
        return float(np.exp(np.min(smoothed_log)))

    raise ValueError(f"Unknown LOSS_MODE: {LOSS_MODE!r}")


def build_loss_lookup(
    records: list[RunRecord],
    row_keys: list[str],
    col_keys: list[str],
) -> dict[tuple[tuple[str, ...], tuple[str, ...]], float | None]:
    lookup = {}

    for rec in records:
        row_t = get_key_tuple(rec, row_keys)
        col_t = get_key_tuple(rec, col_keys)
        cell_key = (row_t, col_t)

        if cell_key in lookup:
            raise ValueError(
                "Multiple runs map to the same grid cell. "
                "Add another varying key to ROW_KEYS or COLUMN_KEYS.\n"
                f"  row keys={row_keys}, row tuple={row_t}\n"
                f"  col keys={col_keys}, col tuple={col_t}\n"
                f"  duplicate run={rec.stem}"
            )

        lookup[cell_key] = read_best_validation_loss(rec.run_dir)

    return lookup


def add_loss_table(
    doc: Document,
    records: list[RunRecord],
    row_keys: list[str],
    col_keys: list[str],
    value_formatters: dict[str, dict[str, str]],
) -> None:
    doc.add_heading(
        f"Rows: {' > '.join(row_keys)} | Columns: {' > '.join(col_keys)}",
        level=2,
    )

    row_tuples = get_unique_axis_tuples(records, row_keys)
    col_tuples = get_unique_axis_tuples(records, col_keys)
    loss_lookup = build_loss_lookup(records, row_keys, col_keys)

    n_row_header_rows = max(1, len(col_keys))
    n_row_label_cols = max(1, len(row_keys))

    n_rows = n_row_header_rows + len(row_tuples)
    n_cols = n_row_label_cols + len(col_tuples)

    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.style = "Table Grid"

    for i, key in enumerate(row_keys):
        table.cell(n_row_header_rows - 1, i).text = key

    for level, col_key in enumerate(col_keys):
        spans = value_repeats_every(col_tuples, level)

        for start, end, val in spans:
            label_val = format_field_value(col_key, val, value_formatters)
            cell = table.cell(level, n_row_label_cols + start)
            cell.text = f"{col_key}={label_val}"

            if end - start > 1:
                cell.merge(table.cell(level, n_row_label_cols + end - 1))

    for i, row_t in enumerate(row_tuples, start=n_row_header_rows):
        for j, value in enumerate(row_t):
            row_key = row_keys[j]
            table.cell(i, j).text = format_field_value(
                row_key,
                value,
                value_formatters,
            )

    for i, row_t in enumerate(row_tuples, start=n_row_header_rows):
        for j, col_t in enumerate(col_tuples, start=n_row_label_cols):
            loss = loss_lookup.get((row_t, col_t))
            table.cell(i, j).text = format_loss_value(loss)

    doc.add_paragraph("")


def main() -> None:
    run_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])

    records = []
    for run_dir in run_dirs:
        rec = parse_run_dir(run_dir, FIG_SUBDIR, FIG_FILES, FIELD_CHUNKS)
        if rec is not None:
            records.append(rec)

    if not records:
        raise RuntimeError("No parseable run directories found.")

    varying_keys = find_varying_keys(records, PARAM_KEYS)

    validate_axis_keys(varying_keys, ROW_KEYS, COLUMN_KEYS)

    row_keys_base = [k for k in ROW_KEYS if k in varying_keys]
    col_keys_base = [k for k in COLUMN_KEYS if k in varying_keys]

    if not row_keys_base and not col_keys_base:
        raise ValueError("No varying keys selected for rows or columns.")

    all_display_keys = row_keys_base + col_keys_base
    value_formatters = build_field_value_formatter(records, all_display_keys)

    print("\nVarying keys:")
    for k in varying_keys:
        vals = sorted(
            {str(r.params.get(k, "")) for r in records},
            key=lambda v: field_value_sort_key(k, v),
        )
        vals = [format_field_value(k, v, value_formatters) for v in vals]
        print(f"  {k}: {vals}")

    doc = Document()
    doc.add_heading("Validation loss tables", level=1)
    doc.add_paragraph(f"Loss file: {LOSS_REL_PATH}")
    doc.add_paragraph(f"Loss column: {LOSS_COLUMN}")
    doc.add_paragraph(f"Loss mode: {LOSS_MODE}")
    if LOSS_MODE == "min_smoothed":
        doc.add_paragraph(f"Smoothing: gaussian_filter1d(log(loss), sigma={SMOOTH_SIGMA})")

    n_tables = 0

    for col_keys in itertools.permutations(col_keys_base):
        for row_keys in itertools.permutations(row_keys_base):
            add_loss_table(
                doc=doc,
                records=records,
                row_keys=list(row_keys),
                col_keys=list(col_keys),
                value_formatters=value_formatters,
            )
            n_tables += 1

    out_docx = ROOT / "report_validation_loss_tables.docx"
    doc.save(out_docx)

    print(f"\nWrote {n_tables} tables.")
    print(f"Wrote: {out_docx}")


if __name__ == "__main__":
    main()
