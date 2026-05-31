from __future__ import annotations

import os
from pathlib import Path

from docx import Document
from docx.shared import Inches
from docx_helper import (
    extract_run_template,
    build_field_chunks,
    RunRecord,
    parse_run_dir,
    find_varying_keys,
    build_combo_dict
    )


ROOT = (
    Path(os.environ["CMB_ML_DATA"]) / 
    "Datasets/I_512_1850/obs2ps_train"
)

DIRNAME_SPEC = (
    "${nn_model}_Nsims-${n_sims}_lr-${learning_rate_final}_lre-${learning_rate_ramp_epochs}_${optimizer}_wd-${adamw_wd}_sps-${scale_ps}_spx-${model.scale_px}_pe-${use_pos_enc_lut}_en-${norm_kind}_${activation}_r1-${down_reductions}_r2-${spatial_reduction}_r3-${face_reduction}_${model_num}/"
)

COLUMN_PRIORITY = [
    "down_reductions",
    "adamw_wd",
    "learning_rate_final",
]

FIELD_VALUE_ORDER = {
    "down_reductions": ["max", "Xm", "XXm", "mXX", "avg"],
}

FIG_SUBDIR = "Analysis_Figures"

FIG_FILES = {
    "loss": "loss_plot.png",
    "train_predictions": "fig_D_predictions_Train_ep_best_peaks.png",
    "valid_predictions": "fig_D_predictions_Valid_ep_best_peaks.png",
    "test_predictions": "fig_D_predictions_Test_ep_best_peaks.png",
    "testFFN_predictions": "fig_D_predictions_TestFFN_ep_best_peaks.png",
    "testFP_predictions": "fig_D_predictions_TestFP_ep_best_peaks.png",
    "planck_predictions": "fig_D_predictions_Test_ep_best_peaks_PLANCK.png",
}

IMAGE_SPECS = [
    ("Loss", "loss"),
    ("Train predictions", "train_predictions"),
    ("Valid predictions", "valid_predictions"),
    ("Test predictions", "test_predictions"),
    ("TestFFN predictions", "testFFN_predictions"),
    ("TestFP predictions", "testFP_predictions"),
    ("Predictions on Planck", "planck_predictions"),
]

RUN_TEMPLATE = extract_run_template(DIRNAME_SPEC)
FIELD_CHUNKS = build_field_chunks(RUN_TEMPLATE)
PARAM_KEYS = [chunk.combined_name for chunk in FIELD_CHUNKS]


def order_varying_keys(
    varying_keys: list[str],
    column_priority: list[str],
) -> list[str]:
    """
    Reorder varying keys using a user-specified priority list.

    Behavior
    --------
    1. Keys in `column_priority` that are present in `varying_keys` are kept,
       in the order given by `column_priority`.
    2. Keys in `varying_keys` that are NOT mentioned in `column_priority`
       are placed at the FRONT of the result.
    3. Raise if `column_priority` contains duplicates.
    4. Warn if `column_priority` contains keys that are not actually varying.
    5. Warn if there are varying keys not mentioned in `column_priority`.
    """

    # --- strict check: duplicates in COLUMN_PRIORITY ---
    seen = set()
    dupes = [k for k in column_priority if k in seen or seen.add(k)]
    if dupes:
        raise ValueError(f"Duplicate keys in COLUMN_PRIORITY: {dupes}")

    varying_set = set(varying_keys)
    priority_set = set(column_priority)

    # --- warn: priority keys not actually varying ---
    extra_in_priority = [k for k in column_priority if k not in varying_set]
    if extra_in_priority:
        print(
            "[warn] COLUMN_PRIORITY contains keys that are not varying: "
            f"{extra_in_priority}"
        )

    # --- warn: varying keys not mentioned ---
    unmentioned_varying = [k for k in varying_keys if k not in priority_set]
    if unmentioned_varying:
        print(
            "[warn] varying keys not mentioned in COLUMN_PRIORITY; "
            "placing them first: "
            f"{unmentioned_varying}"
        )

    # --- final ordering ---
    ordered_prioritized = [k for k in column_priority if k in varying_set]

    return unmentioned_varying + ordered_prioritized

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


def record_sort_key(rec: RunRecord, keys: list[str]):
    return tuple(
        field_value_sort_key(key, str(rec.params.get(key, "")))
        for key in keys
    )


def format_field_value(
    key: str,
    value: str,
    value_formatters: dict[str, dict[str, str]],
) -> str:
    value = str(value)
    return value_formatters.get(key, {}).get(value, value)


def build_docx(records: list[RunRecord], varying_keys: list[str], image_specs, out_path: Path):
    doc = Document()
    doc.add_heading("Training run report", level=1)

    value_formatters = build_field_value_formatter(records, varying_keys)
    records = sorted(records, key=lambda rec: record_sort_key(rec, varying_keys))

    n_cols = len(varying_keys) + len(image_specs)

    table = doc.add_table(rows=1, cols=n_cols)
    table.style = "Table Grid"

    hdr = table.rows[0].cells

    # parameter columns first
    for j, key in enumerate(varying_keys):
        hdr[j].text = key

    # then image columns
    for j, (title, _) in enumerate(image_specs, start=len(varying_keys)):
        hdr[j].text = title

    for rec in records:
        row_cells = table.add_row().cells

        # parameter values
        for j, key in enumerate(varying_keys):
            raw_value = str(rec.params.get(key, ""))
            row_cells[j].text = format_field_value(key, raw_value, value_formatters)

        # images
        for j, (_, fig_key) in enumerate(image_specs, start=len(varying_keys)):
            p = rec.get_figure(fig_key)
            if p is not None and Path(p).exists():
                paragraph = row_cells[j].paragraphs[0]
                run = paragraph.add_run()
                run.add_picture(str(p), width=Inches(10.0))
            else:
                row_cells[j].text = ""

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

    run_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])

    records = []
    for run_dir in run_dirs:
        rec = parse_run_dir(run_dir, FIG_SUBDIR, FIG_FILES, FIELD_CHUNKS)
        if rec is not None:
            records.append(rec)

    if not records:
        raise RuntimeError("No parseable run directories found.")

    varying_keys = find_varying_keys(records, PARAM_KEYS)
    varying_keys = order_varying_keys(varying_keys, COLUMN_PRIORITY)

    value_formatters = build_field_value_formatter(records, varying_keys)

    print("\nVarying keys:")
    for k in varying_keys:
        vals = sorted(
            {str(r.params.get(k, "")) for r in records},
            key=lambda v: field_value_sort_key(k, v),
        )
        vals = [format_field_value(k, v, value_formatters) for v in vals]
        print(f"  {k}: {vals}")

    combo_dict = build_combo_dict(records, varying_keys)
    print(f"\ncombo_dict has {len(combo_dict)} entries")

    out_docx = ROOT / "report_training_runs.docx"
    build_docx(records, varying_keys, IMAGE_SPECS, out_docx)
    print(f"\nWrote: {out_docx}")


if __name__ == "__main__":
    main()
