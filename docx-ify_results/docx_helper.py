from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


TOKEN_RE = re.compile(r"\$\{([^}]+)\}")


def normalize_field_name(field_name: str) -> str:
    return field_name.replace(".", "_")


def extract_run_template(dirname_spec: str) -> str:
    return dirname_spec.strip("/").split("/")[-1]


@dataclass
class FieldChunk:
    field_names_raw: list[str]
    literal_before: str
    literal_after: str

    @property
    def field_names(self) -> list[str]:
        return [normalize_field_name(f) for f in self.field_names_raw]

    @property
    def combined_name(self) -> str:
        return "_".join(self.field_names)


def build_field_chunks(template: str) -> list[FieldChunk]:
    """
    Parse the template into chunks. Consecutive adjacent ${...}${...} fields
    are merged into one chunk.

    Example:
        "..._${basis_kind}${degree}_${model_num}"
    becomes chunks roughly like:
        [FieldChunk(["basis_kind", "degree"], literal_before="_", literal_after="_"),
         FieldChunk(["model_num"], literal_before="_", literal_after="")]
    """
    matches = list(TOKEN_RE.finditer(template))
    if not matches:
        return []

    chunks: list[FieldChunk] = []

    i = 0
    while i < len(matches):
        m0 = matches[i]
        field_names_raw = [m0.group(1)]

        literal_before = template[matches[i - 1].end():m0.start()] if i > 0 else template[:m0.start()]

        j = i
        while j + 1 < len(matches) and matches[j].end() == matches[j + 1].start():
            j += 1
            field_names_raw.append(matches[j].group(1))

        literal_after = (
            template[matches[j].end():matches[j + 1].start()]
            if j + 1 < len(matches)
            else template[matches[j].end():]
        )

        chunks.append(
            FieldChunk(
                field_names_raw=field_names_raw,
                literal_before=literal_before,
                literal_after=literal_after,
            )
        )
        i = j + 1

    return chunks


@dataclass
class RunRecord:
    run_dir: Path
    stem: str
    params: dict[str, str]
    figures: dict[str, Path | None]

    def __getattr__(self, name: str):
        try:
            return self.params[name]
        except KeyError:
            raise AttributeError(name)

    def get_figure(self, key: str) -> Path | None:
        return self.figures.get(key)


def parse_run_name(name: str, field_chunks) -> dict[str, str]:
    pos = 0
    out: dict[str, str] = {}

    for idx, chunk in enumerate(field_chunks):
        if not name.startswith(chunk.literal_before, pos):
            got = name[pos:pos + max(20, len(chunk.literal_before))]
            raise ValueError(
                f"Expected literal_before {chunk.literal_before!r} "
                f"for chunk {chunk.combined_name!r} at position {pos}, got {got!r}"
            )
        pos += len(chunk.literal_before)

        if chunk.literal_after:
            end = name.find(chunk.literal_after, pos)
            if end == -1:
                raise ValueError(
                    f"Could not find literal_after {chunk.literal_after!r} "
                    f"for chunk {chunk.combined_name!r}"
                )
            value = name[pos:end]
            pos = end
        else:
            value = name[pos:]
            pos = len(name)

        out[chunk.combined_name] = value

    if pos != len(name):
        raise ValueError(f"Unparsed trailing text: {name[pos:]!r}")

    return out


def parse_run_dir(
    run_dir: Path,
    fig_subdir,
    fig_files,
    field_chunks,
) -> Optional[RunRecord]:
    try:
        params = parse_run_name(run_dir.name, field_chunks)
    except Exception as e:
        print(f"[warn] could not parse: {run_dir.name}")
        print(f"       reason: {e}")
        return None

    fig_dir = run_dir / fig_subdir

    figures = {
        fig_key: (fig_dir / filename if filename is not None else None)
        for fig_key, filename in fig_files.items()
    }

    return RunRecord(
        run_dir=run_dir,
        stem=run_dir.name,
        params=params,
        figures=figures,
    )


def find_varying_keys(records: list[RunRecord], param_keys) -> list[str]:
    varying = []
    for key in param_keys:
        vals = {r.params.get(key) for r in records}
        if len(vals) > 1:
            varying.append(key)
    return varying


def build_combo_dict(
    records: list[RunRecord],
    key_order: list[str],
) -> dict[tuple, RunRecord]:
    out = {}
    for r in records:
        k = tuple(r.params.get(key) for key in key_order)
        out[k] = r
    return out


def format_record_label(rec: RunRecord, keys: list[str]) -> str:
    return "\n".join(f"{key}={rec.params.get(key)}" for key in keys)
