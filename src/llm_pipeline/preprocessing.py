from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .labels import LABEL_PRIORITY, LABEL_TO_STATE, require_label


@dataclass
class RecordingConfig:
    annotation_id: str
    participant_no: int
    midge_id: int | str
    video_name: str
    annotation_json: str | Path
    acc_paths: list[str | Path]
    video_path: str | Path | None
    source_video_start_abs_time: str | pd.Timestamp
    clip_start_offset_s: float
    clip_end_offset_s: float
    target_fs: float = 50.0

    @property
    def video_start_abs_time(self) -> pd.Timestamp:
        source_start = pd.Timestamp(self.source_video_start_abs_time)
        return source_start + pd.to_timedelta(float(self.clip_start_offset_s), unit="s")

    @property
    def video_end_abs_time(self) -> pd.Timestamp:
        source_start = pd.Timestamp(self.source_video_start_abs_time)
        return source_start + pd.to_timedelta(float(self.clip_end_offset_s), unit="s")

    @property
    def duration_s(self) -> float:
        return float(self.clip_end_offset_s) - float(self.clip_start_offset_s)


@dataclass
class ManualLabelResult:
    manual_df: pd.DataFrame
    manual_csv: Path
    config_json: Path
    summary_json: Path


def _as_path_list(paths: Iterable[str | Path]) -> list[Path]:
    return [Path(path) for path in paths]


def load_acc_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ACC CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    df = df.drop(columns=[col for col in df.columns if not col or col.lower().startswith("unnamed")], errors="ignore")
    df = df.rename(columns={col: col.lower() for col in df.columns})
    if "timestamp" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    if "datetime" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"datetime": "time"})

    required = {"time", "x", "y", "z"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"ACC CSV missing required columns {sorted(missing)}: {path}")

    out = df[["time", "x", "y", "z"]].copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    for axis in ["x", "y", "z"]:
        out[axis] = pd.to_numeric(out[axis], errors="coerce")
    out = out.dropna(subset=["time", "x", "y", "z"])
    return out.sort_values("time").reset_index(drop=True)


def load_and_resample_acc(acc_paths: Iterable[str | Path], target_fs: float = 50.0) -> pd.DataFrame:
    paths = _as_path_list(acc_paths)
    if not paths:
        raise ValueError("At least one ACC CSV path is required.")

    frames = [load_acc_csv(path) for path in paths]
    acc = pd.concat(frames, ignore_index=True)
    acc = acc.groupby("time", as_index=False)[["x", "y", "z"]].mean().sort_values("time")
    if acc.empty:
        raise ValueError("No valid ACC rows were loaded.")

    rule = pd.to_timedelta(1.0 / float(target_fs), unit="s")
    resampled = (
        acc.set_index("time")[["x", "y", "z"]]
        .resample(rule)
        .mean()
        .interpolate(method="time")
        .dropna()
        .reset_index()
    )
    return resampled


def load_covfee_segments(annotation_json: str | Path) -> pd.DataFrame:
    path = Path(annotation_json)
    if not path.exists():
        raise FileNotFoundError(f"COVFEE annotation JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("annotations", payload.get("segments", []))
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Unsupported COVFEE annotation JSON shape: {path}")

    df = pd.DataFrame(rows)
    required = {"start", "end", "category"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"COVFEE annotation JSON missing fields {sorted(missing)}: {path}")

    df = df[["start", "end", "category"]].copy()
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df["manual_label"] = df["category"].map(require_label)
    df = df.dropna(subset=["start", "end", "manual_label"]).sort_values(["start", "end"]).reset_index(drop=True)
    invalid = df[df["end"] < df["start"]]
    if not invalid.empty:
        raise ValueError(f"Annotation contains segments with end < start: {path}")
    return df


def _config_to_jsonable(config: RecordingConfig) -> dict[str, object]:
    data = asdict(config)
    data["annotation_json"] = str(config.annotation_json)
    data["acc_paths"] = [str(path) for path in config.acc_paths]
    data["video_path"] = None if config.video_path is None else str(config.video_path)
    data["source_video_start_abs_time"] = str(pd.Timestamp(config.source_video_start_abs_time))
    data["video_start_abs_time"] = str(config.video_start_abs_time)
    data["video_end_abs_time"] = str(config.video_end_abs_time)
    data["duration_s"] = config.duration_s
    return data


def build_manual_labels_from_covfee(config: RecordingConfig, output_root: str | Path) -> ManualLabelResult:
    if config.duration_s <= 0:
        raise ValueError("clip_end_offset_s must be greater than clip_start_offset_s.")

    output_root = Path(output_root)
    manual_dir = output_root / "manual_labels"
    manual_dir.mkdir(parents=True, exist_ok=True)

    acc = load_and_resample_acc(config.acc_paths, target_fs=config.target_fs)
    video_start = config.video_start_abs_time
    video_end = config.video_end_abs_time
    video_acc = acc[(acc["time"] >= video_start) & (acc["time"] <= video_end)].copy().reset_index(drop=True)
    if video_acc.empty:
        raise ValueError(
            "No ACC samples overlap the configured video interval. "
            f"Check SOURCE_VIDEO_START_ABS_TIME and clip offsets: {video_start} -> {video_end}"
        )

    segments = load_covfee_segments(config.annotation_json)
    manual = video_acc.copy()
    manual["annotation_id"] = config.annotation_id
    manual["participant_no"] = int(config.participant_no)
    manual["midge_id"] = str(config.midge_id)
    manual["video_name"] = config.video_name
    manual["relative_time_s"] = (manual["time"] - video_start).dt.total_seconds()
    manual["state"] = LABEL_TO_STATE["Still"]
    manual["manual_label"] = "Still"

    priority_buffer = np.full(len(manual), LABEL_PRIORITY["Still"], dtype=np.int16)
    for row in segments.itertuples(index=False):
        label = str(row.manual_label)
        abs_start = video_start + pd.to_timedelta(float(row.start), unit="s")
        abs_end = video_start + pd.to_timedelta(float(row.end), unit="s")
        abs_start = max(abs_start, video_start)
        abs_end = min(abs_end, video_end)
        if abs_end < abs_start:
            continue

        mask = (manual["time"] >= abs_start) & (manual["time"] <= abs_end)
        if not mask.any():
            continue
        mask_np = mask.to_numpy()
        update_mask = mask_np & (LABEL_PRIORITY[label] >= priority_buffer)
        if update_mask.any():
            priority_buffer[update_mask] = LABEL_PRIORITY[label]
            manual.loc[update_mask, "state"] = LABEL_TO_STATE[label]
            manual.loc[update_mask, "manual_label"] = label

    manual = manual[
        [
            "annotation_id",
            "participant_no",
            "midge_id",
            "video_name",
            "time",
            "relative_time_s",
            "x",
            "y",
            "z",
            "state",
            "manual_label",
        ]
    ]

    manual_csv = manual_dir / f"manual_labels_{config.annotation_id}.csv"
    config_json = manual_dir / f"recording_config_{config.annotation_id}.json"
    summary_json = manual_dir / f"manual_label_summary_{config.annotation_id}.json"

    manual.to_csv(manual_csv, index=False)
    config_json.write_text(json.dumps(_config_to_jsonable(config), indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "annotation_id": config.annotation_id,
        "n_samples": int(len(manual)),
        "target_fs": float(config.target_fs),
        "video_start_abs_time": str(video_start),
        "video_end_abs_time": str(video_end),
        "label_counts": manual["manual_label"].value_counts().sort_index().to_dict(),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return ManualLabelResult(manual, manual_csv, config_json, summary_json)
