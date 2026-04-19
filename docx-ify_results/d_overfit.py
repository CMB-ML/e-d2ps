from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Inches

from docx_helper import (
    extract_run_template,
    build_field_chunks,
    parse_run_dir,
    find_varying_keys,
    RunRecord,
)


ROOT = Path(
    "/CMB_Data/Datasets/I_128_20850/working_directory"
)

DIRNAME_SPEC = (
    "${features_type}2ps_${nn_model}_polyfit_overfit_sweep1/"
    "Nsims-${n_sims}_lr-${learning_rate}_${optimizer}_wd-${adamw_wd}"
    "_sps-${model.scale_ps}_spx-${model.scale_px}_pe-${use_pos_enc_lut}"
    "_en-${norm_kind}_${activation}_r1-${down_reductions}"
    "_r2-${spatial_reduction}_${basis_kind}${degree}_${model_num}/"
)

ROW_PRIORITY = [
    "basis_kind_degree",
]

COLUMN_PRIORITY = [
    "learning_rate",
]

FIG_SUBDIR = "Analysis_Figures"
FIG_FILES = {
    "loss": "loss_plot.png",
}

IMAGE_SPECS = [
    ("Loss", "loss"),
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


def tuple_sort_key(t: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("" if x is None else str(x) for x in t)


def format_tuple_label(keys: list[str], values: tuple[str, ...]) -> str:
    if not keys:
        return ""
    if len(keys) == 1:
        return str(values[0])
    return "\n".join(f"{k}={v}" for k, v in zip(keys, values))


def put_image_or_blank(cell, path: Path | None, img_width_in: float = 10.0) -> None:
    if path is not None and Path(path).exists():
        paragraph = cell.paragraphs[0]
        run = paragraph.add_run()
        run.add_picture(str(path), width=Inches(img_width_in))
    else:
        cell.text = ""


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
    doc = Document()
    doc.add_heading("Overfit run report", level=1)

    row_tuples = sorted(
        {get_key_tuple(rec, row_keys) for rec in records},
        key=tuple_sort_key,
    )
    col_tuples = sorted(
        {get_key_tuple(rec, col_keys) for rec in records},
        key=tuple_sort_key,
    )

    grid = build_grid_lookup(records, row_keys, col_keys)

    n_row_header_cols = 1
    n_cols = n_row_header_cols + len(col_tuples)

    table = doc.add_table(rows=1 + len(row_tuples), cols=n_cols)
    table.style = "Table Grid"

    hdr = table.rows[0].cells
    hdr[0].text = format_tuple_label(row_keys, tuple(row_keys))

    for j, col_t in enumerate(col_tuples, start=1):
        hdr[j].text = format_tuple_label(col_keys, col_t)

    fig_key = IMAGE_SPECS[0][1]

    for i, row_t in enumerate(row_tuples, start=1):
        row_cells = table.rows[i].cells
        row_cells[0].text = format_tuple_label(row_keys, row_t)

        for j, col_t in enumerate(col_tuples, start=1):
            rec = grid.get((row_t, col_t))
            p = None if rec is None else rec.get_figure(fig_key)
            put_image_or_blank(row_cells[j], p, img_width_in=10.0)

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


def main() -> None:
    validate_image_specs(FIG_FILES, IMAGE_SPECS)
    if len(IMAGE_SPECS) != 1:
        raise ValueError(
            f"Overfit grid expects exactly one image spec, got {len(IMAGE_SPECS)}"
        )

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
        vals = sorted({getattr(r, k) for r in records}, key=str)
        print(f"  {k}: {vals}")

    print("\nResolved axes:")
    print(f"  rows:    {row_keys}")
    print(f"  columns: {col_keys}")

    out_docx = ROOT / "report_overfit_runs.docx"
    build_docx_grid(records, row_keys, col_keys, out_docx)
    print(f"\nWrote: {out_docx}")


if __name__ == "__main__":
    main()
