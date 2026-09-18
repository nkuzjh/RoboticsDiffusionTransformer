"""Qualitative Seen-10 localization panels.

Each map panel contains the published radar on the left and a vertical strip
of the corresponding FPV frames on the right.  The selection is made from
dataset rows with a local ``random.Random(seed)`` instance, so it is stable
across dataloader order, model checkpoints and process state.  Prediction
rows are joined by the shared evaluator identity (``map/file_frame`` or bare
``file_frame``) and GT stays in dataset metadata only.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from data.csgo_seen10 import SEEN_MAPS, Seen10Dataset


SAMPLE_COLORS = (
    (255, 72, 72),
    (56, 220, 95),
    (72, 132, 255),
    (255, 218, 48),
    (255, 75, 220),
    (30, 225, 225),
    (255, 145, 40),
    (178, 105, 255),
    (245, 245, 245),
    (100, 255, 185),
)
PREDICTION_FIELDS = ("pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw")


def render_localization(
    dataset: Seen10Dataset,
    predictions: Sequence[Mapping[str, Any]] | str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    seed: int = 0,
    per_map: int = 10,
    *,
    radar_size: int = 1800,
    fpv_width: int = 480,
) -> list[Path]:
    """Render one fixed-sample localization panel per Seen-10 map.

    ``predictions`` may be the already parsed JSONL row list or a path to a
    ``predictions.jsonl`` file/directory.  Each map independently receives up
    to ``per_map`` rows selected with ``seed``.  The selected sample IDs are
    written to ``visualization_manifest.json`` beside the PNG panels.

    Predictions are expected in normalized Benchmark v2 pose space.  Their
    XY marker coordinates are clamped only for display; values shown in the
    FPV header are converted to physical coordinates using the dataset's
    published Z ranges.
    """

    if not isinstance(dataset, Seen10Dataset):
        # Duck typing is useful for tiny test datasets, but fail with a useful
        # message when a caller accidentally passes a batch instead.
        if not hasattr(dataset, "rows") or not hasattr(dataset, "pose") or not hasattr(dataset, "physical_pose"):
            raise TypeError("dataset must expose rows, pose() and physical_pose()")
    if not isinstance(per_map, int) or per_map <= 0:
        raise ValueError(f"per_map must be a positive integer, got {per_map!r}")
    if per_map > len(SAMPLE_COLORS):
        raise ValueError(f"At most {len(SAMPLE_COLORS)} visualization samples are supported")
    if not isinstance(radar_size, int) or radar_size <= 0 or not isinstance(fpv_width, int) or fpv_width <= 0:
        raise ValueError("radar_size and fpv_width must be positive integers")

    records = list(getattr(dataset, "rows", getattr(dataset, "records", ())))
    record_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        map_name = str(record.get("map_name", record.get("map", "")))
        sample_id = _sample_id_for_record(record, map_name)
        identity = (map_name, sample_id)
        if identity in record_by_identity:
            raise ValueError(f"Duplicate dataset identity: {identity}")
        if map_name not in SEEN_MAPS:
            raise ValueError(f"Dataset contains a map outside Seen-10: {map_name!r}")
        record_by_identity[identity] = record
        grouped[map_name].append(record)

    prediction_rows = _read_predictions(predictions)
    predictions_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for index, prediction in enumerate(prediction_rows):
        if not isinstance(prediction, Mapping):
            raise ValueError(f"Prediction row {index} must be an object")
        map_name = prediction.get("map_name", prediction.get("map"))
        if not isinstance(map_name, str) or map_name not in SEEN_MAPS:
            raise ValueError(f"Prediction row {index} has an invalid map_name: {map_name!r}")
        sample_id = _sample_id_for_prediction(prediction, map_name)
        identity = (map_name, sample_id)
        if identity in predictions_by_identity:
            raise ValueError(f"Duplicate prediction identity: {identity}")
        values: list[float] = []
        for field in PREDICTION_FIELDS:
            try:
                value = float(prediction[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Prediction {identity} needs numeric {field}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Prediction {identity} has non-finite {field}: {value}")
            values.append(value)
        normalized = dict(prediction)
        normalized.update({field: value for field, value in zip(PREDICTION_FIELDS, values)})
        predictions_by_identity[identity] = normalized

    rng = random.Random(int(seed))
    selected_by_map: dict[str, list[Mapping[str, Any]]] = {}
    for map_name in SEEN_MAPS:
        candidates = grouped[map_name]
        count = min(per_map, len(candidates))
        selected_by_map[map_name] = rng.sample(candidates, count)

    manifest = {
        "seed": int(seed),
        "per_map": int(per_map),
        "pose_order": "xyzhw = x, y, z, pitch, yaw; angles shown in degrees",
        "prediction_pose_space": "normalized",
        "maps": {
            map_name: [_sample_id_for_record(row, map_name) for row in selected_by_map[map_name]]
            for map_name in SEEN_MAPS
        },
    }
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "visualization_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != manifest:
            raise ValueError(f"Visualization selection differs from existing {manifest_path}")
        expected_outputs = [output_root / f"{map_name}.png" for map_name in SEEN_MAPS if selected_by_map[map_name]]
        if all(path.is_file() for path in expected_outputs):
            return expected_outputs

    body_font = _font(max(11, min(16, radar_size // max(1, per_map) // 12)))
    row_height = max(1, radar_size // max(1, per_map))
    panel_height = row_height * max(1, per_map)
    # Keep one square radar; the published radar artwork is a top-down map and
    # markers use the normalized x/y contract directly.
    map_size = panel_height
    marker_margin = max(20, map_size // 70)
    outputs: list[Path] = []

    for map_name in SEEN_MAPS:
        selected = selected_by_map[map_name]
        if not selected:
            continue
        identities = [(map_name, _sample_id_for_record(row, map_name)) for row in selected]
        missing = [identity for identity in identities if identity not in predictions_by_identity]
        if missing:
            raise ValueError(f"Missing visualization predictions: {missing[:5]}")
        records_for_map = selected
        radar_path = Path(records_for_map[0]["radar_path"])
        with Image.open(radar_path) as source:
            radar = source.convert("RGB").resize((map_size, map_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (map_size + fpv_width, panel_height), "black")
        canvas.paste(radar, (0, 0))
        radar_draw = ImageDraw.Draw(canvas)
        _draw_radar_title(radar_draw, map_name, body_font, map_size)

        for index, record in enumerate(records_for_map):
            identity = (map_name, _sample_id_for_record(record, map_name))
            prediction = predictions_by_identity[identity]
            color = SAMPLE_COLORS[index]
            gt_norm = [float(value) for value in dataset.pose(record)]
            pred_norm = [float(prediction[field]) for field in PREDICTION_FIELDS]
            if not all(math.isfinite(value) for value in gt_norm + pred_norm):
                raise ValueError(f"Non-finite pose for visualization {identity}")
            gt_xy = _map_pixel(gt_norm[0], gt_norm[1], map_size, marker_margin)
            pred_xy = _map_pixel(pred_norm[0], pred_norm[1], map_size, marker_margin)
            radar_draw.line((gt_xy, pred_xy), fill=color, width=max(3, map_size // 600))

            gt_radius = max(8, map_size // 100)
            radar_draw.ellipse(
                (gt_xy[0] - gt_radius, gt_xy[1] - gt_radius, gt_xy[0] + gt_radius, gt_xy[1] + gt_radius),
                fill=color,
                outline=(0, 0, 0),
                width=max(2, map_size // 600),
            )
            pred_radius = max(gt_radius + 5, map_size // 75)
            radar_draw.ellipse(
                (pred_xy[0] - pred_radius, pred_xy[1] - pred_radius, pred_xy[0] + pred_radius, pred_xy[1] + pred_radius),
                outline=(0, 0, 0),
                width=max(5, map_size // 260),
            )
            radar_draw.ellipse(
                (pred_xy[0] - pred_radius, pred_xy[1] - pred_radius, pred_xy[0] + pred_radius, pred_xy[1] + pred_radius),
                outline=color,
                width=max(3, map_size // 360),
            )

            fpv = _fit_fpv(record["image_path"], (fpv_width, row_height))
            fpv_draw = ImageDraw.Draw(fpv, "RGBA")
            text_height = max(42, min(row_height, row_height // 2))
            fpv_draw.rectangle((0, 0, fpv_width, text_height), fill=(0, 0, 0, 175))
            dot_radius = max(6, min(12, row_height // 12))
            fpv_draw.ellipse(
                (7, 7, 7 + 2 * dot_radius, 7 + 2 * dot_radius),
                fill=color + (255,),
                outline=(0, 0, 0, 255),
                width=2,
            )
            gt_physical = dataset.physical_pose(gt_norm, map_name)
            pred_physical = dataset.physical_pose(pred_norm, map_name)
            gt_label = "gt_xyzhw [" + ",".join(f"{value:.1f}" for value in gt_physical) + "]"
            pred_label = "pred_xyzhw [" + ",".join(f"{value:.1f}" for value in pred_physical) + "]"
            text_center = fpv_width / 2
            _draw_text(fpv_draw, (text_center, 4), gt_label, body_font, anchor="mt")
            _draw_text(fpv_draw, (text_center, 24), pred_label, body_font, anchor="mt")
            canvas.paste(fpv, (map_size, index * row_height))

        output_path = output_root / f"{map_name}.png"
        if not output_path.exists():
            temporary = output_root / f".{map_name}.png.tmp"
            canvas.save(temporary, format="PNG", optimize=True)
            os.replace(temporary, output_path)
        outputs.append(output_path)

    if not manifest_path.exists():
        temporary = output_root / ".visualization_manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    return outputs


def _read_predictions(predictions: Sequence[Mapping[str, Any]] | str | os.PathLike[str]) -> list[Mapping[str, Any]]:
    if isinstance(predictions, (str, os.PathLike)):
        path = Path(predictions).expanduser()
        if path.is_dir():
            path = path / "predictions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Localization predictions not found: {path}")
        rows: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid prediction JSON at {path}:{line_no}: {exc}") from exc
                rows.append(row)
        return rows
    return list(predictions)


def _map_name(record: Mapping[str, Any]) -> str:
    return str(record.get("map_name", record.get("map", "")))


def _sample_id_for_record(record: Mapping[str, Any], map_name: str) -> str:
    sample_id = record.get("sample_id")
    if isinstance(sample_id, str) and sample_id:
        return _file_frame(sample_id, map_name)
    file_frame = record.get("file_frame")
    if not isinstance(file_frame, str) or not file_frame:
        raise ValueError(f"Dataset row has no file_frame/sample_id: {record!r}")
    return Path(file_frame).stem


def _sample_id_for_prediction(prediction: Mapping[str, Any], map_name: str) -> str:
    sample_id = prediction.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"Prediction for {map_name} requires sample_id")
    return _file_frame(sample_id, map_name)


def _file_frame(sample_id: str, map_name: str) -> str:
    parts = sample_id.replace("\\", "/").split("/")
    if len(parts) > 1 and parts[-2] != map_name:
        raise ValueError(f"sample_id map does not match map_name: {sample_id!r}, {map_name!r}")
    frame = Path(parts[-1]).stem
    if not frame:
        raise ValueError(f"Invalid sample_id: {sample_id!r}")
    return frame


def _map_pixel(x: float, y: float, size: int, margin: int) -> tuple[float, float]:
    return (
        min(max(float(x) * size, margin), size - margin - 1),
        min(max(float(y) * size, margin), size - margin - 1),
    )


def _fit_fpv(path: str | os.PathLike[str], size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        source = source.convert("RGB")
        contained = ImageOps.contain(source, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def _font(size: int, *, bold: bool = False):
    filename = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    for directory in (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/dejavu")):
        path = directory / filename
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: Any, anchor: str = "la") -> None:
    draw.text(xy, text, font=font, fill="white", stroke_width=2, stroke_fill="black", anchor=anchor)


def _draw_radar_title(draw: ImageDraw.ImageDraw, map_name: str, font: Any, size: int) -> None:
    _draw_text(draw, (12, 10), map_name, font, anchor="la")


__all__ = ["PREDICTION_FIELDS", "SAMPLE_COLORS", "render_localization"]
