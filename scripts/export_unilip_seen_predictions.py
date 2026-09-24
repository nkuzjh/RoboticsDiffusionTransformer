#!/usr/bin/env python3
"""Export UniLIP pred_norm JSONL to the unmodified Seen-10 evaluator contract.

Only sample identity and pred_norm are consumed. GT fields in source records
are deliberately ignored. The source epsilon is explicit, never inferred from
test errors or fitted from labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

FIELDS = ("pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw")
SEEN_MAPS = {"cs_agency", "cs_italy", "de_ancient", "de_anubis", "de_dust2",
             "de_inferno", "de_mirage", "de_nuke", "de_overpass", "de_train"}


def convert_row(row: dict, z_ranges: dict, epsilon: float) -> dict:
    name = row.get("map", row.get("map_name"))
    identity = row.get("file_frame", row.get("sample_id"))
    if name not in SEEN_MAPS or name not in z_ranges or not isinstance(identity, str):
        raise ValueError("Missing/invalid Seen-10 identity or calibration")
    identity = identity.removesuffix(".jpg").removesuffix(".png")
    if "/" in identity:
        prefix, identity = identity.split("/", 1)
        if prefix != name:
            raise ValueError("sample ID map disagrees with map field")
    import re
    if re.fullmatch(r"file_num\d+_frame_\d+", identity) is None:
        raise ValueError(f"Invalid file_frame: {identity!r}")
    values = row.get("pred_norm")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError("Expected pred_norm to be a flat five-element list")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Nonfinite/nonnumeric predicted pose")
    values = list(map(float, values))
    span = float(z_ranges[name]["z_max"]) - float(z_ranges[name]["z_min"])
    if not math.isfinite(span) or span <= 0 or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Invalid Z span or source epsilon")
    values[2] *= (span + epsilon) / span
    return {"map_name": name, "sample_id": identity, **dict(zip(FIELDS, values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UniLIP JSONL containing map, file_frame, pred_norm")
    parser.add_argument("--output", type=Path, required=True, help="New standard predictions.jsonl")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--source-z-epsilon", type=float, required=True, help="exp32/exp32_loc use 1e-6")
    args = parser.parse_args()
    metadata_path = args.output.with_name(args.output.name + ".export.json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError("Export destination already exists; choose a new output")
    calibration = json.loads(args.calibration.read_text())
    rows, ids = [], set()
    for line_number, line in enumerate(args.input.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = convert_row(json.loads(line), calibration["z_ranges"], args.source_z_epsilon)
        key = row["map_name"], row["sample_id"]
        if key in ids:
            raise ValueError(f"Duplicate ID on line {line_number}: {key}")
        ids.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("No predictions found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    metadata_path.write_text(json.dumps({
        "input": str(args.input.resolve()), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "source_z_epsilon": args.source_z_epsilon, "target_z_epsilon": 0,
        "rows": len(rows), "uses_ground_truth": False, "clamp": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
