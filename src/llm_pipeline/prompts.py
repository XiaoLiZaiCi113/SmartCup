from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import LABEL_ORDER


UPRIGHT_GRAVITY_REFERENCE = (0.0, 0.0, -1.0)
DEFAULT_METHODS = [
    "llm_zero_shot_xyz_timeline",
    "llm_few_shot_xyz_timeline",
    "llm_zero_shot_xyz_rag_tilt",
    "llm_few_shot_xyz_rag_tilt",
]

METHOD_DISPLAY_NAMES = {
    "llm_zero_shot_xyz_timeline": "Zero-shot xyz timeline",
    "llm_few_shot_xyz_timeline": "Few-shot xyz timeline",
    "llm_zero_shot_xyz_rag_tilt": "Zero-shot xyz RAG tilt",
    "llm_few_shot_xyz_rag_tilt": "Few-shot xyz RAG tilt",
}


ZERO_SHOT_TIMELINE_PROMPT = """You are an expert of smart-cup accelerometer based human activity recognition.

The device is a smart cup held by the user. The sensor contains a three-axis accelerometer only, sampled at 50 Hz in the smart-cup coordinate frame.

Your task is to localize activities in one complete recording. The input is not divided into windows. Treat the JSON as one continuous timeline from 0.00 seconds to duration_s.

Candidate classes, use these exact labels only:
- Still
- Gesture
- Drinking
- Toasting
- Nodding

Use only the complete xyz timeline as evidence. If the signal contains multiple activity patterns, split the full timeline into non-overlapping temporal segments. Segments must be in chronological order and should cover the full interval [0.00, duration_s]. Use Still for inactive or uncertain intervals.

For each segment, return start_time, end_time, predicted_class, and reason. Use seconds relative to the beginning of the complete recording.

Return valid JSON only, exactly one object:
{"annotation_id":"<annotation_id>","segments":[{"start_time":0.00,"end_time":1.24,"predicted_class":"<class>","reason":"<brief explanation>"}]}"""


FEW_SHOT_TIMELINE_PROMPT = """You are an expert of smart-cup accelerometer based human activity recognition.

The device is a smart cup held by the user. The sensor contains a three-axis accelerometer only, sampled at 50 Hz in the smart-cup coordinate frame.

Your task is to localize activities in one complete candidate recording by comparing it with labeled reference samples from the same smart-cup setup.

Candidate classes, use these exact labels only:
- Still
- Gesture
- Drinking
- Toasting
- Nodding

Input JSON fields:
- reference_samples: manually labeled example segments
- candidate_timeline: the complete recording to localize

Use the reference_samples as the primary evidence. Compare patterns in the candidate timeline with the labeled examples. Do not invent new classes. Split the full timeline into non-overlapping chronological temporal segments and use Still for inactive or uncertain intervals.

For each segment, return start_time, end_time, predicted_class, and reason. Use seconds relative to the beginning of the complete candidate recording.

Return valid JSON only, exactly one object:
{"annotation_id":"<annotation_id>","segments":[{"start_time":0.00,"end_time":1.24,"predicted_class":"<class>","reason":"<brief explanation>"}]}"""


ZERO_SHOT_WINDOW_PROMPT = """You are an expert of IMU-based human activity analysis.

The IMU data is collected from a smart cup held by the user with a sampling rate of 50 Hz. The data is represented in the smart-cup coordinate frame. For each window, the three-axis accelerations, derived tilt angle relative to the upright smart-cup gravity reference [0, 0, -1], and RAG-style segment summaries for full/start/mid/end are given in the accompanying JSON input.

The person's action belongs to one of the following categories:
- Still
- Gesture
- Drinking
- Toasting
- Nodding

Within each window, identify one or more temporal segments. For each segment, return start_time, end_time, predicted_class, and reason. Use start_time and end_time in seconds relative to the beginning of the current window.

Return valid JSON only, exactly one array:
[{"window_id":0,"segments":[{"start_time":0.00,"end_time":0.42,"predicted_class":"<class>","reason":"<brief explanation>"}]}]"""


FEW_SHOT_WINDOW_PROMPT = """You are an expert of smart-cup accelerometer based human activity recognition.

The device is a smart cup held by the user. The sensor contains a three-axis accelerometer only, sampled at 50 Hz in the smart-cup coordinate frame.

Your task is to localize activities inside each candidate window by comparing the candidate window with labeled reference samples from the same smart-cup setup.

Candidate classes, use these exact labels only:
- Still
- Gesture
- Drinking
- Toasting
- Nodding

Input JSON fields:
- split: describes how reference_samples and candidate_windows were selected
- reference_samples: manually labeled reference segments with raw xyz sequences and summaries
- candidate_windows: windows to localize, each with raw xyz sequence, tilt angle sequence, and segment_statistics

Use the reference_samples as the primary evidence. Compare each candidate window with the manually labeled reference samples using the raw sequences and statistical summaries. If a candidate window contains multiple activity patterns, split it into temporal segments and assign each segment the closest matching reference-supported class.

Return valid JSON only, exactly one array:
[{"window_id":0,"segments":[{"start_time":0.00,"end_time":0.42,"predicted_class":"<class>","reason":"<brief explanation>"}]}]"""


@dataclass
class PromptBundleResult:
    output_root: Path
    annotation_id: str
    window_index_csv: Path
    generated_files: list[Path]
    prediction_root: Path
    method_artifacts: dict[str, dict[str, Path]]


def tilt_angle_deg(gx, gy, gz, ref_vector=UPRIGHT_GRAVITY_REFERENCE) -> np.ndarray:
    gref = np.asarray(ref_vector, dtype=float)
    gref_norm = np.linalg.norm(gref)
    g = np.stack([gx, gy, gz], axis=1).astype(float)
    g_norm = np.linalg.norm(g, axis=1)
    denom = g_norm * gref_norm
    cosv = np.divide(g @ gref, denom, out=np.ones_like(g_norm, dtype=float), where=denom > 0)
    return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))


def summarize_sequence(values) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": None, "max": None, "min": None, "q1": None, "q3": None, "std": None, "median": None}
    return {
        "mean": round(float(np.mean(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "q1": round(float(np.percentile(arr, 25)), 6),
        "q3": round(float(np.percentile(arr, 75)), 6),
        "std": round(float(np.std(arr, ddof=0)), 6),
        "median": round(float(np.median(arr)), 6),
    }


def three_way_slices(length: int) -> dict[str, slice]:
    edges = np.linspace(0, length, 4, dtype=int)
    return {"start": slice(edges[0], edges[1]), "mid": slice(edges[1], edges[2]), "end": slice(edges[2], edges[3])}


def _round_list(values, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def _write_json(path: Path, payload: object, compact: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"ensure_ascii": False}
    if compact:
        kwargs["separators"] = (",", ":")
    else:
        kwargs["indent"] = 2
    path.write_text(json.dumps(payload, **kwargs), encoding="utf-8")


def build_timeline_payload(manual_df: pd.DataFrame, annotation_id: str, sampling_rate_hz: float = 50.0) -> dict[str, object]:
    frame = manual_df.sort_values("time").reset_index(drop=True)
    timestamps = pd.to_datetime(frame["time"])
    n_samples = len(frame)
    duration_s = round(n_samples / float(sampling_rate_hz), 6)
    return {
        "annotation_id": annotation_id,
        "sampling_rate_hz": float(sampling_rate_hz),
        "classes": LABEL_ORDER,
        "time_axis": {
            "start_timestamp": timestamps.iloc[0].isoformat() if n_samples else None,
            "end_timestamp": timestamps.iloc[-1].isoformat() if n_samples else None,
            "n_samples": int(n_samples),
            "duration_s": duration_s,
            "relative_time_s": [round(index / float(sampling_rate_hz), 6) for index in range(n_samples)],
        },
        "raw_xyz_sequence": {
            "x": _round_list(frame["x"]),
            "y": _round_list(frame["y"]),
            "z": _round_list(frame["z"]),
        },
    }


def _window_statistics(window_df: pd.DataFrame, tilt_values: np.ndarray) -> dict[str, dict[str, dict[str, float | None]]]:
    values = {
        "x": window_df["x"].to_numpy(dtype=float),
        "y": window_df["y"].to_numpy(dtype=float),
        "z": window_df["z"].to_numpy(dtype=float),
        "tilt": np.asarray(tilt_values, dtype=float),
    }
    segment_slices = {"full": slice(0, len(window_df))}
    segment_slices.update(three_way_slices(len(window_df)))
    return {
        name: {axis: summarize_sequence(axis_values[segment_slice]) for axis, axis_values in values.items()}
        for name, segment_slice in segment_slices.items()
    }


def build_window_payloads(
    manual_df: pd.DataFrame,
    annotation_id: str,
    window_seconds: float = 10.0,
    sampling_rate_hz: float = 50.0,
    min_tail_seconds: float = 1.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    frame = manual_df.sort_values("time").reset_index(drop=True)
    frame["time"] = pd.to_datetime(frame["time"])
    window_size = max(1, int(round(float(window_seconds) * float(sampling_rate_hz))))
    min_tail_size = max(1, int(round(float(min_tail_seconds) * float(sampling_rate_hz))))
    tilt = tilt_angle_deg(frame["x"].to_numpy(), frame["y"].to_numpy(), frame["z"].to_numpy())

    payloads: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    window_id = 0
    start = 0
    while start < len(frame):
        end = min(start + window_size, len(frame))
        if end - start < min_tail_size and payloads:
            break
        window_df = frame.iloc[start:end].copy().reset_index(drop=True)
        window_tilt = tilt[start:end]
        start_time = pd.Timestamp(window_df["time"].iloc[0])
        end_time = pd.Timestamp(window_df["time"].iloc[-1]) + pd.to_timedelta(1.0 / float(sampling_rate_hz), unit="s")
        payloads.append(
            {
                "annotation_id": annotation_id,
                "window_id": int(window_id),
                "window_duration_s": round(len(window_df) / float(sampling_rate_hz), 6),
                "raw_xyz_sequence": {
                    "x": _round_list(window_df["x"]),
                    "y": _round_list(window_df["y"]),
                    "z": _round_list(window_df["z"]),
                },
                "tilt_angle_deg_sequence": _round_list(window_tilt),
                "segment_statistics": _window_statistics(window_df, window_tilt),
            }
        )
        index_rows.append(
            {
                "annotation_id": annotation_id,
                "window_id": int(window_id),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "start_idx_in_recording": int(start),
                "end_idx_in_recording": int(end - 1),
                "start_s": round(start / float(sampling_rate_hz), 6),
                "end_s": round(end / float(sampling_rate_hz), 6),
            }
        )
        window_id += 1
        start = end

    return payloads, pd.DataFrame(index_rows)


def build_reference_samples(
    manual_df: pd.DataFrame,
    max_per_class: int = 8,
    reference_train_ratio: float = 0.70,
    window_seconds: float = 10.0,
    sampling_rate_hz: float = 50.0,
) -> list[dict[str, object]]:
    frame = manual_df.sort_values("time").reset_index(drop=True).copy()
    frame["time"] = pd.to_datetime(frame["time"])
    train_end = int(len(frame) * float(reference_train_ratio))
    train = frame.iloc[:train_end].copy().reset_index(drop=True)
    if train.empty:
        return []

    tilt = tilt_angle_deg(train["x"].to_numpy(), train["y"].to_numpy(), train["z"].to_numpy())
    train["tilt"] = tilt
    chunk_size = max(1, int(round(float(window_seconds) * float(sampling_rate_hz))))
    samples: list[dict[str, object]] = []
    run_start = 0
    reference_id = 0

    for index in range(1, len(train) + 1):
        if index == len(train) or train.loc[index, "manual_label"] != train.loc[index - 1, "manual_label"]:
            label = str(train.loc[index - 1, "manual_label"])
            for chunk_start in range(run_start, index, chunk_size):
                chunk_end = min(chunk_start + chunk_size, index)
                chunk = train.iloc[chunk_start:chunk_end].copy().reset_index(drop=True)
                if chunk.empty:
                    continue
                raw_xyz = {axis: _round_list(chunk[axis]) for axis in ["x", "y", "z"]}
                sample = {
                    "reference_id": f"ref_{reference_id:04d}",
                    "label": label,
                    "start_time": pd.Timestamp(chunk["time"].iloc[0]).isoformat(),
                    "end_time": pd.Timestamp(chunk["time"].iloc[-1]).isoformat(),
                    "duration_s": round(len(chunk) / float(sampling_rate_hz), 6),
                    "raw_xyz_sequence": raw_xyz,
                    "tilt_angle_deg_sequence": _round_list(chunk["tilt"]),
                    "segment_statistics": _window_statistics(chunk, chunk["tilt"].to_numpy(dtype=float)),
                }
                samples.append(sample)
                reference_id += 1
            run_start = index

    balanced: list[dict[str, object]] = []
    for label in LABEL_ORDER:
        label_samples = [sample for sample in samples if sample["label"] == label]
        if len(label_samples) <= max_per_class:
            balanced.extend(label_samples)
        else:
            keep = np.linspace(0, len(label_samples) - 1, max_per_class, dtype=int)
            balanced.extend(label_samples[int(i)] for i in keep)
    return balanced


def _prediction_dirs(prediction_root: Path, annotation_id: str, method_id: str, n_runs: int) -> list[Path]:
    dirs = []
    for run_index in range(1, int(n_runs) + 1):
        run_dir = prediction_root / annotation_id / method_id / f"run_{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        hint = run_dir / "PUT_LLM_OUTPUT_DATA_JSON_HERE.txt"
        if not hint.exists():
            hint.write_text("Save the LLM response for this run as data.json in this folder.\n", encoding="utf-8")
        dirs.append(run_dir)
    return dirs


def write_prompt_bundle(
    manual_df: pd.DataFrame,
    output_root: str | Path,
    annotation_id: str,
    *,
    n_runs: int = 4,
    window_seconds: float = 10.0,
    sampling_rate_hz: float = 50.0,
    reference_train_ratio: float = 0.70,
    few_shot_candidate_start_ratio: float = 0.0,
) -> PromptBundleResult:
    output_root = Path(output_root)
    prompt_root = output_root / "prompts"
    input_root = output_root / "input_json"
    prediction_root = output_root / "llm_predictions"
    recording_input_root = input_root / annotation_id
    recording_input_root.mkdir(parents=True, exist_ok=True)

    timeline_payload = build_timeline_payload(manual_df, annotation_id, sampling_rate_hz=sampling_rate_hz)
    window_payloads, window_index = build_window_payloads(
        manual_df,
        annotation_id,
        window_seconds=window_seconds,
        sampling_rate_hz=sampling_rate_hz,
    )
    window_index_csv = recording_input_root / f"window_index_{annotation_id}.csv"
    window_index.to_csv(window_index_csv, index=False)

    reference_samples = build_reference_samples(
        manual_df,
        reference_train_ratio=reference_train_ratio,
        window_seconds=window_seconds,
        sampling_rate_hz=sampling_rate_hz,
    )
    candidate_start_s = float(timeline_payload["time_axis"]["duration_s"]) * float(few_shot_candidate_start_ratio)
    candidate_window_ids = set(window_index.loc[window_index["start_s"] >= candidate_start_s, "window_id"].astype(int))
    candidate_windows = [payload for payload in window_payloads if int(payload["window_id"]) in candidate_window_ids]

    payloads = {
        "llm_zero_shot_xyz_timeline": timeline_payload,
        "llm_few_shot_xyz_timeline": {
            "split": {
                "reference_train_ratio": float(reference_train_ratio),
                "reference_source": "first part of the sample-level manual-label timeline",
                "candidate_source": "complete candidate timeline",
            },
            "reference_samples": reference_samples,
            "candidate_timeline": timeline_payload,
        },
        "llm_zero_shot_xyz_rag_tilt": window_payloads,
        "llm_few_shot_xyz_rag_tilt": {
            "split": {
                "reference_train_ratio": float(reference_train_ratio),
                "candidate_start_ratio": float(few_shot_candidate_start_ratio),
                "reference_source": "first part of the sample-level manual-label timeline",
                "candidate_source": "candidate windows whose start time is at or after candidate_start_ratio",
            },
            "reference_samples": reference_samples,
            "candidate_windows": candidate_windows,
        },
    }
    prompts = {
        "llm_zero_shot_xyz_timeline": ZERO_SHOT_TIMELINE_PROMPT.replace("<annotation_id>", annotation_id),
        "llm_few_shot_xyz_timeline": FEW_SHOT_TIMELINE_PROMPT.replace("<annotation_id>", annotation_id),
        "llm_zero_shot_xyz_rag_tilt": ZERO_SHOT_WINDOW_PROMPT,
        "llm_few_shot_xyz_rag_tilt": FEW_SHOT_WINDOW_PROMPT,
    }

    generated_files: list[Path] = [window_index_csv]
    method_artifacts: dict[str, dict[str, Path]] = {}
    for method_id in DEFAULT_METHODS:
        method_prompt_dir = prompt_root / annotation_id / method_id
        method_input_dir = input_root / annotation_id / method_id
        prompt_txt = method_prompt_dir / "prompt.txt"
        input_json = method_input_dir / "input.json"
        prompt_txt.parent.mkdir(parents=True, exist_ok=True)
        prompt_txt.write_text(prompts[method_id].strip() + "\n", encoding="utf-8")
        _write_json(input_json, payloads[method_id], compact=True)
        run_dirs = _prediction_dirs(prediction_root, annotation_id, method_id, n_runs=n_runs)
        method_artifacts[method_id] = {
            "prompt_txt": prompt_txt,
            "input_json": input_json,
            "prediction_dir": prediction_root / annotation_id / method_id,
        }
        generated_files.extend([prompt_txt, input_json, *run_dirs])

    return PromptBundleResult(
        output_root=output_root,
        annotation_id=annotation_id,
        window_index_csv=window_index_csv,
        generated_files=generated_files,
        prediction_root=prediction_root,
        method_artifacts=method_artifacts,
    )
