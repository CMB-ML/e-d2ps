from __future__ import annotations

from pathlib import Path
import os

from docx import Document
from docx.shared import Inches

from docx_helper import (
    extract_run_template,
    build_field_chunks,
    parse_run_dir,
    find_varying_keys,
    RunRecord,
)

REPORT_NAME = "report_plots.docx"
# REPORT_NAME = "report_best_pred_plots.docx"
# REPORT_NAME = "report_train_rho_plots.docx"
# REPORT_NAME = "report_train_FVE_plots.docx"

ROOT = (
    Path(os.environ["CMB_ML_DATA"]) / 
    "Datasets/I_8_Inf_CMB_only/otf_cmb2ps_fcn_scales"
)

DIRNAME_SPEC = (
    "pred_${spectrum_type}-lr_${learning_rate}-lbl_${prep_label}-ftr_${prep_feature}-fcn_str_${fcn_string}-lmax_${lmax}-sigma_scale${scale}/"
)

ROW_PRIORITY = [
    "fcn_string",
    "spectrum_type"
]

COLUMN_PRIORITY = [
    "scale",
]

ROW_PRIORITY = [
    "adamw_wd",
    "learning_rate"
]

FIELD_VALUE_ORDER = {
    "fcn_string": ["N_10aR_O", "N_10aR_10aR_O", "N_20aR_O", "N_20aR_20aR_O", "N_50aR_O", "N_50aR_50aR_O", "N_100aR_O", "N_100aR_100aR_O", "N_200aR_O", "N_200aR_200aR_O", "N_500aR_O", "N_500aR_500aR_O", "N_1000aR_O", "N_1000aR_1000aR_O"],
}

FIG_SUBDIR = "Analysis_Figures"
FIG_FILES = {
    "loss": "loss_plot.png",
    "preds": "fig_D_predictions_Valid_ep_best.png",
    "rho": "fig_E_train_rho.png",
    "FVE": "fig_E_train_FVE.png",
}

IMAGE_SPECS = [
    ("Loss", "loss"),
    ("Preds", "preds"),
    ("rho", "rho"),
    ("FVE", "FVE"),
]

RUN_TEMPLATE = extract_run_template(DIRNAME_SPEC)
FIELD_CHUNKS = build_field_chunks(RUN_TEMPLATE)
PARAM_KEYS = [chunk.combined_name for chunk in FIELD_CHUNKS]


def _find_dupes(keys: list[str]) -> list[str]:
    seen = set()
    dupes = []
    for k in keys:
        if k in seen and k not in dupes:
            dupes.append(k)
        seen.add(k)
    return dupes


def resolve_axis_keys(
    varying_keys: list[str],
    row_priority: list[str],
    column_priority: list[str],
) -> tuple[list[str], list[str]]:
    row_dupes = _find_dupes(row_priority)
    if row_dupes:
        raise ValueError(f"Duplicate keys in ROW_PRIORITY: {row_dupes}")

    col_dupes = _find_dupes(column_priority)
    if col_dupes:
        raise ValueError(f"Duplicate keys in COLUMN_PRIORITY: {col_dupes}")

    overlap = [k for k in row_priority if k in set(column_priority)]
    if overlap:
        raise ValueError(
            f"Keys appear in both ROW_PRIORITY and COLUMN_PRIORITY: {overlap}"
        )

    varying_set = set(varying_keys)
    row_set = set(row_priority)
    col_set = set(column_priority)
    assigned_set = row_set | col_set

    extra_rows = [k for k in row_priority if k not in varying_set]
    if extra_rows:
        print(f"[warn] ROW_PRIORITY contains keys that are not varying: {extra_rows}")

    extra_cols = [k for k in column_priority if k not in varying_set]
    if extra_cols:
        print(f"[warn] COLUMN_PRIORITY contains keys that are not varying: {extra_cols}")

    unassigned = [k for k in varying_keys if k not in assigned_set]
    if unassigned:
        print(
            "[warn] varying keys not assigned to rows/columns; "
            f"placing them first in rows: {unassigned}"
        )

    ordered_rows = unassigned + [k for k in row_priority if k in varying_set]
    ordered_cols = [k for k in column_priority if k in varying_set]

    if not ordered_rows and not ordered_cols:
        raise ValueError("No varying keys were assigned to rows or columns.")

    return ordered_rows, ordered_cols


def get_key_tuple(rec: RunRecord, keys: list[str]) -> tuple[str, ...]:
    return tuple(str(rec.params.get(k, "")) for k in keys)


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
    # Use general format first (removes trailing zeros)
    s = f"{x:.15g}"

    # If it didn’t choose scientific notation, force it
    if "e" not in s.lower():
        s = f"{x:.15g}".lower()
        s = f"{float(s):.15e}"

    mant, exp = s.lower().split("e")

    # Clean mantissa (strip trailing zeros and dot)
    mant = mant.rstrip("0").rstrip(".")

    # Normalize exponent (remove leading zeros)
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


def axis_tuple_sort_key(keys: list[str], t: tuple[str, ...]):
    return tuple(
        field_value_sort_key(key, value)
        for key, value in zip(keys, t)
    )


def format_axis_value(
    key: str,
    value: str,
    value_formatters: dict[str, dict[str, str]],
) -> str:
    value = str(value)
    return value_formatters.get(key, {}).get(value, value)


# def tuple_sort_key(t: tuple[str, ...]) -> tuple[str, ...]:
#     return tuple("" if x is None else str(x) for x in t)


# def format_tuple_label(keys: list[str], values: tuple[str, ...]) -> str:
#     if not keys:
#         return ""
#     if len(keys) == 1:
#         return str(values[0])
#     return "\n".join(f"{k}={v}" for k, v in zip(keys, values))


def put_image_or_blank(cell, path: Path | None, img_width_in: float = 10.0) -> None:
    if path is not None and Path(path).exists():
        paragraph = cell.paragraphs[0]
        run = paragraph.add_run()
        run.add_picture(str(path), width=Inches(img_width_in))
    else:
        cell.text = ""


# def get_unique_axis_tuples(
#     records: list[RunRecord],
#     keys: list[str],
# ) -> list[tuple[str, ...]]:
#     return sorted(
#         {get_key_tuple(rec, keys) for rec in records},
#         key=tuple_sort_key,
#     )

def get_unique_axis_tuples(
    records: list[RunRecord],
    keys: list[str],
) -> list[tuple[str, ...]]:
    return sorted(
        {get_key_tuple(rec, keys) for rec in records},
        key=lambda t: axis_tuple_sort_key(keys, t),
    )

def value_repeats_every(col_tuples: list[tuple[str, ...]], level: int) -> list[tuple[int, int, str]]:
    """
    For a given column-key level, return runs of identical values across columns.

    Returns a list of:
        (start_col_index, end_col_index_exclusive, label)

    Example:
        col_tuples = [
            ("C.1", "D.1"),
            ("C.1", "D.2"),
            ("C.2", "D.1"),
            ("C.2", "D.2"),
        ]

        level=0 -> [(0, 2, "C=C.1"), (2, 4, "C=C.2")]
        level=1 -> [(0, 1, "D=D.1"), (1, 2, "D=D.2"), ...]
    """
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


def build_grid_lookup(
    records: list[RunRecord],
    row_keys: list[str],
    col_keys: list[str],
) -> dict[tuple[tuple[str, ...], tuple[str, ...]], RunRecord]:
    grid = {}

    for rec in records:
        row_t = get_key_tuple(rec, row_keys)
        col_t = get_key_tuple(rec, col_keys)
        cell_key = (row_t, col_t)

        if cell_key in grid:
            raise ValueError(
                "Multiple runs map to the same grid cell:\n"
                f"  row keys={row_keys}, row tuple={row_t}\n"
                f"  col keys={col_keys}, col tuple={col_t}\n"
                f"  first run={grid[cell_key].stem}\n"
                f"  second run={rec.stem}"
            )

        grid[cell_key] = rec

    return grid


def build_docx_grid(
    records: list[RunRecord],
    row_keys: list[str],
    col_keys: list[str],
    out_path: Path,
):
    if len(IMAGE_SPECS) != 1:
        raise ValueError(
            f"Overfit grid expects exactly one image spec, got {len(IMAGE_SPECS)}"
        )

    fig_key = IMAGE_SPECS[0][1]
    value_formatters = build_field_value_formatter(records, row_keys + col_keys)

    doc = Document()
    doc.add_heading("Overfit run report", level=1)

    row_tuples = get_unique_axis_tuples(records, row_keys)
    col_tuples = get_unique_axis_tuples(records, col_keys)

    grid = build_grid_lookup(records, row_keys, col_keys)

    n_row_header_rows = max(1, len(col_keys))
    n_row_label_cols = max(1, len(row_keys))

    n_rows = n_row_header_rows + len(row_tuples)
    n_cols = n_row_label_cols + len(col_tuples)

    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.style = "Table Grid"

    # ----------------------------
    # Header block, top-left corner
    # ----------------------------
    for i, key in enumerate(row_keys):
        table.cell(n_row_header_rows - 1, i).text = key

    # ----------------------------
    # Column header rows
    # ----------------------------
    # One header row per column key.
    # Row 0 = major column key, next row = next column key, etc.
    for level, col_key in enumerate(col_keys):
        spans = value_repeats_every(col_tuples, level)

        for start, end, val in spans:
            cell = table.cell(level, n_row_label_cols + start)
            # cell.text = f"{col_key}={val}"
            cell.text = f"{col_key}={format_axis_value(col_key, val, value_formatters)}"

            # Merge horizontally if the same value spans multiple columns
            if end - start > 1:
                cell.merge(table.cell(level, n_row_label_cols + end - 1))

    # ----------------------------
    # Left row-label columns
    # ----------------------------
    for i, row_t in enumerate(row_tuples, start=n_row_header_rows):
        for j, value in enumerate(row_t):
            # table.cell(i, j).text = str(value)
            row_key = row_keys[j]
            table.cell(i, j).text = format_axis_value(row_key, value, value_formatters)

    # ----------------------------
    # Interior cells: figures
    # ----------------------------
    for i, row_t in enumerate(row_tuples, start=n_row_header_rows):
        for j, col_t in enumerate(col_tuples, start=n_row_label_cols):
            rec = grid.get((row_t, col_t))
            p = None if rec is None else rec.get_figure(fig_key)
            put_image_or_blank(table.cell(i, j), p, img_width_in=10.0)

    doc.save(out_path)


def validate_image_specs(fig_files: dict[str, str], image_specs: list[tuple[str, str]]) -> None:
    fig_keys = set(fig_files.keys())
    spec_keys = [key for _, key in image_specs]

    missing = [key for key in spec_keys if key not in fig_keys]
    if missing:
        raise ValueError(
            "IMAGE_SPECS refers to figure keys not present in FIG_FILES: "
            f"{missing}"
        )


def add_grid_table(
    doc: Document,
    records: list[RunRecord],
    row_keys: list[str],
    col_keys: list[str],
    fig_label: str,
    fig_key: str,
) -> None:
    """Add one labelled table to an existing Document object."""
    value_formatters = build_field_value_formatter(records, row_keys + col_keys)

    doc.add_heading(fig_label, level=2)

    row_tuples = get_unique_axis_tuples(records, row_keys)
    col_tuples = get_unique_axis_tuples(records, col_keys)

    grid = build_grid_lookup(records, row_keys, col_keys)

    n_row_header_rows = max(1, len(col_keys))
    n_row_label_cols  = max(1, len(row_keys))
    n_rows = n_row_header_rows + len(row_tuples)
    n_cols = n_row_label_cols  + len(col_tuples)

    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.style = "Table Grid"

    for i, key in enumerate(row_keys):
        table.cell(n_row_header_rows - 1, i).text = key

    for level, col_key in enumerate(col_keys):
        for start, end, val in value_repeats_every(col_tuples, level):
            cell = table.cell(level, n_row_label_cols + start)
            cell.text = f"{col_key}={format_axis_value(col_key, val, value_formatters)}"
            if end - start > 1:
                cell.merge(table.cell(level, n_row_label_cols + end - 1))

    for i, row_t in enumerate(row_tuples, start=n_row_header_rows):
        for j, value in enumerate(row_t):
            table.cell(i, j).text = format_axis_value(row_keys[j], value, value_formatters)

    for i, row_t in enumerate(row_tuples, start=n_row_header_rows):
        for j, col_t in enumerate(col_tuples, start=n_row_label_cols):
            rec = grid.get((row_t, col_t))
            p   = None if rec is None else rec.get_figure(fig_key)
            put_image_or_blank(table.cell(i, j), p, img_width_in=10.0)


def build_docx_all(
    records: list[RunRecord],
    row_keys: list[str],
    col_keys: list[str],
    out_path: Path,
) -> None:
    doc = Document()
    doc.add_heading("Overfit run report", level=1)

    for fig_label, fig_key in IMAGE_SPECS:
        add_grid_table(doc, records, row_keys, col_keys, fig_label, fig_key)
        doc.add_page_break()

    doc.save(out_path)


def main() -> None:
    validate_image_specs(FIG_FILES, IMAGE_SPECS)

    run_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])

    records = []
    for run_dir in run_dirs:
        rec = parse_run_dir(run_dir, FIG_SUBDIR, FIG_FILES, FIELD_CHUNKS)
        if rec is not None:
            records.append(rec)

    if not records:
        raise RuntimeError("No parseable run directories found.")

    varying_keys = find_varying_keys(records, PARAM_KEYS)

    row_keys, col_keys = resolve_axis_keys(
        varying_keys,
        ROW_PRIORITY,
        COLUMN_PRIORITY,
    )

    print("\nVarying keys:")
    for k in varying_keys:
        # vals = sorted({getattr(r, k) for r in records}, key=str)
        vals = sorted(
            {str(r.params.get(k, "")) for r in records},
            key=lambda v: field_value_sort_key(k, v),
        )
        vals = [
            format_axis_value(k, v, build_field_value_formatter(records, [k]))
            for v in vals
        ]
        print(f"  {k}: {vals}")

    print("\nResolved axes:")
    print(f"  rows:    {row_keys}")
    print(f"  columns: {col_keys}")

    out_docx = ROOT / REPORT_NAME
    # build_docx_grid(records, row_keys, col_keys, out_docx)
    build_docx_all(records, row_keys, col_keys, out_docx)
    print(f"\nWrote: {out_docx}")


if __name__ == "__main__":
    main()
