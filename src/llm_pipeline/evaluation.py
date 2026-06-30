from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_recall_fscore_support

from .labels import ACTIVE_LABELS, LABEL_ORDER, LABEL_PRIORITY, coarse_label
from .plotting import render_metrics_table_png, sample_end_times, save_combined_timeline, save_repeated_run_timeline, save_timeline_plot
from .prompts import DEFAULT_METHODS, METHOD_DISPLAY_NAMES


SAMPLING_RATE_HZ = 50.0
TIOU_THRESHOLDS = np.array([0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)
REPORT_METRICS = ["macro_precision", "macro_recall", "macro_f1", "sample_accuracy", "active_accuracy", "mAP"]


@dataclass
class EvaluationResult:
    figure_dir: Path
    model_metrics_csv: Path
    model_metrics_png: Path
    combined_timeline_png: Path | None
    missing_predictions_csv: Path | None
    summary: pd.DataFrame


def load_manual_labels(manual_label_csv: str | Path) -> pd.DataFrame:
    path = Path(manual_label_csv)
    if not path.exists():
        raise FileNotFoundError(f"Manual-label CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"annotation_id", "time", "x", "y", "z", "manual_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manual-label CSV missing fields {sorted(missing)}: {path}")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def segments_from_llm_json(prediction_path: str | Path) -> list[dict]:
    prediction_path = Path(prediction_path)
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "segments" in payload:
            segments = payload.get("segments", [])
        elif "predictions" in payload:
            return _segments_from_window_items(payload.get("predictions", []))
        else:
            raise ValueError(f"Timeline prediction object must contain 'segments': {prediction_path}")
    elif isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "window_id" in payload[0]:
            return _segments_from_window_items(payload)
        segments = payload
    else:
        raise ValueError(f"Unsupported LLM prediction JSON shape: {prediction_path}")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"No segments found in prediction JSON: {prediction_path}")
    return [segment for segment in segments if isinstance(segment, dict)]


def _segments_from_window_items(items: object) -> list[dict]:
    if not isinstance(items, list):
        return []
    rows: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "segments" in item and isinstance(item["segments"], list):
            for segment in item["segments"]:
                if isinstance(segment, dict):
                    out = dict(segment)
                    out["window_id"] = item.get("window_id")
                    rows.append(out)
        else:
            rows.append(dict(item))
    return rows


def _sample_slice_from_seconds(start_s: float, end_s: float, n_samples: int, sampling_rate_hz: float) -> slice | None:
    start_idx = max(0, int(math.floor(float(start_s) * float(sampling_rate_hz))))
    end_idx = min(n_samples, int(math.ceil(float(end_s) * float(sampling_rate_hz))))
    if end_idx <= start_idx:
        return None
    return slice(start_idx, end_idx)


def _apply_segment(
    predicted: np.ndarray,
    priority_buffer: np.ndarray,
    seg_slice: slice,
    label: str,
) -> None:
    idx = np.arange(len(predicted))[seg_slice]
    update_idx = idx[LABEL_PRIORITY[label] >= priority_buffer[idx]]
    if update_idx.size:
        priority_buffer[update_idx] = LABEL_PRIORITY[label]
        predicted[update_idx] = label


def rasterize_llm_timeline(
    prediction_path: str | Path,
    manual_df: pd.DataFrame,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
) -> np.ndarray:
    predicted = np.full(len(manual_df), "Still", dtype=object)
    priority_buffer = np.full(len(manual_df), LABEL_PRIORITY["Still"], dtype=np.int16)
    for segment in segments_from_llm_json(prediction_path):
        start_s = float(segment.get("start_time", segment.get("start", 0.0)))
        end_s = float(segment.get("end_time", segment.get("end", start_s)))
        if end_s < start_s:
            continue
        label = coarse_label(segment.get("predicted_class", segment.get("label", "Still")))
        if label is None:
            continue
        seg_slice = _sample_slice_from_seconds(start_s, end_s, len(manual_df), sampling_rate_hz)
        if seg_slice is not None:
            _apply_segment(predicted, priority_buffer, seg_slice, label)
    return predicted


def prediction_df_from_window_json(prediction_path: str | Path, window_index_csv: str | Path) -> pd.DataFrame:
    prediction_path = Path(prediction_path)
    window_index = pd.read_csv(window_index_csv)
    window_index["start_time"] = pd.to_datetime(window_index["start_time"])
    window_index["end_time"] = pd.to_datetime(window_index["end_time"])

    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]
    if not isinstance(payload, list):
        raise ValueError(f"Window prediction JSON must be an array or predictions object: {prediction_path}")

    rows = _segments_from_window_items(payload)
    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise ValueError(f"No window prediction rows found: {prediction_path}")

    if "start" in pred_df.columns and "start_time" not in pred_df.columns:
        pred_df = pred_df.rename(columns={"start": "start_time"})
    if "end" in pred_df.columns and "end_time" not in pred_df.columns:
        pred_df = pred_df.rename(columns={"end": "end_time"})
    required = {"window_id", "start_time", "end_time", "predicted_class"}
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"Window prediction JSON missing fields {sorted(missing)}: {prediction_path}")

    pred_df["window_id"] = pd.to_numeric(pred_df["window_id"], errors="coerce").astype("Int64")
    pred_df["segment_start_s"] = pd.to_numeric(pred_df["start_time"], errors="coerce")
    pred_df["segment_end_s"] = pd.to_numeric(pred_df["end_time"], errors="coerce")
    pred_df["coarse_label"] = pred_df["predicted_class"].map(coarse_label)
    pred_df = pred_df.dropna(subset=["window_id", "segment_start_s", "segment_end_s", "coarse_label"])
    pred_df = pred_df[pred_df["segment_end_s"] >= pred_df["segment_start_s"]].copy()
    if pred_df.empty:
        raise ValueError(f"No valid window prediction intervals after validation: {prediction_path}")

    pred_df["window_id"] = pred_df["window_id"].astype(int)
    pred_df = pred_df.merge(
        window_index[["window_id", "start_time", "end_time"]].rename(
            columns={"start_time": "window_start_time", "end_time": "window_end_time"}
        ),
        on="window_id",
        how="left",
    )
    pred_df = pred_df.dropna(subset=["window_start_time", "window_end_time"]).copy()
    pred_df["abs_start_time"] = pred_df["window_start_time"] + pd.to_timedelta(pred_df["segment_start_s"], unit="s")
    pred_df["abs_end_time"] = pred_df["window_start_time"] + pd.to_timedelta(pred_df["segment_end_s"], unit="s")
    pred_df["abs_start_time"] = pred_df[["abs_start_time", "window_start_time"]].max(axis=1)
    pred_df["abs_end_time"] = pred_df[["abs_end_time", "window_end_time"]].min(axis=1)
    pred_df = pred_df[pred_df["abs_end_time"] >= pred_df["abs_start_time"]].copy()
    return pred_df.sort_values(["window_id", "abs_start_time", "abs_end_time"]).reset_index(drop=True)


def rasterize_llm_windows(
    prediction_path: str | Path,
    manual_df: pd.DataFrame,
    window_index_csv: str | Path,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
) -> np.ndarray:
    pred_df = prediction_df_from_window_json(prediction_path, window_index_csv)
    predicted = np.full(len(manual_df), "Still", dtype=object)
    priority_buffer = np.full(len(manual_df), LABEL_PRIORITY["Still"], dtype=np.int16)
    origin = pd.to_datetime(manual_df["time"]).iloc[0]
    for row in pred_df.itertuples(index=False):
        label = str(row.coarse_label)
        start_s = float((pd.Timestamp(row.abs_start_time) - origin).total_seconds())
        end_s = float((pd.Timestamp(row.abs_end_time) - origin).total_seconds())
        seg_slice = _sample_slice_from_seconds(start_s, end_s, len(manual_df), sampling_rate_hz)
        if seg_slice is not None:
            _apply_segment(predicted, priority_buffer, seg_slice, label)
    return predicted


def eval_table(annotation_id: str, manual_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    n = min(len(manual_df), len(y_pred))
    frame = manual_df.iloc[:n].copy().reset_index(drop=True)
    times = pd.to_datetime(frame["time"])
    return pd.DataFrame(
        {
            "annotation_id": annotation_id,
            "start_time": times,
            "end_time": sample_end_times(times),
            "y_true": frame["manual_label"].astype(str),
            "y_pred": y_pred[:n].astype(str),
        }
    )


def _runs_as_segments(eval_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    frame = eval_df.copy().reset_index(drop=True)
    frame["start_time"] = pd.to_datetime(frame["start_time"])
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    rows: list[dict[str, object]] = []
    start_idx = 0
    labels = frame[label_col].astype(str).tolist()
    for index in range(1, len(frame) + 1):
        if index == len(frame) or labels[index] != labels[index - 1]:
            label = labels[index - 1]
            if label != "Still":
                rows.append(
                    {
                        "video-id": str(frame.loc[start_idx, "annotation_id"]),
                        "t-start": float((frame.loc[start_idx, "start_time"] - frame.loc[0, "start_time"]).total_seconds()),
                        "t-end": float((frame.loc[index - 1, "end_time"] - frame.loc[0, "start_time"]).total_seconds()),
                        "label_name": label,
                        "score": 1.0,
                    }
                )
            start_idx = index
    return pd.DataFrame(rows, columns=["video-id", "t-start", "t-end", "label_name", "score"])


def temporal_iou(target_segment: np.ndarray, candidate_segments: np.ndarray) -> np.ndarray:
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    intersection = np.clip(tt2 - tt1, 0.0, None)
    union = candidate_segments[:, 1] - candidate_segments[:, 0] + target_segment[1] - target_segment[0] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)


def interpolated_prec_rec(precision: np.ndarray, recall: np.ndarray) -> float:
    mprec = np.hstack([[0.0], precision, [0.0]])
    mrec = np.hstack([[0.0], recall, [1.0]])
    for index in range(len(mprec) - 1)[::-1]:
        mprec[index] = max(mprec[index], mprec[index + 1])
    changed = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[changed] - mrec[changed - 1]) * mprec[changed]))


def average_precision_for_class(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    tiou_thresholds: np.ndarray = TIOU_THRESHOLDS,
) -> np.ndarray:
    ap = np.zeros(len(tiou_thresholds), dtype=float)
    if ground_truth.empty or prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth)), dtype=float) * -1
    prediction = prediction.iloc[prediction["score"].to_numpy(dtype=float).argsort()[::-1]].reset_index(drop=True)
    tp = np.zeros((len(tiou_thresholds), len(prediction)), dtype=float)
    fp = np.zeros((len(tiou_thresholds), len(prediction)), dtype=float)
    ground_truth_by_video = ground_truth.groupby("video-id")

    for pred_idx, this_pred in prediction.iterrows():
        try:
            video_gt = ground_truth_by_video.get_group(this_pred["video-id"])
        except KeyError:
            fp[:, pred_idx] = 1
            continue
        this_gt = video_gt.reset_index()
        tiou_values = temporal_iou(
            this_pred[["t-start", "t-end"]].to_numpy(dtype=float),
            this_gt[["t-start", "t-end"]].to_numpy(dtype=float),
        )
        for threshold_idx, threshold in enumerate(tiou_thresholds):
            for gt_idx in tiou_values.argsort()[::-1]:
                if tiou_values[gt_idx] < threshold:
                    fp[threshold_idx, pred_idx] = 1
                    break
                original_gt_idx = int(this_gt.loc[gt_idx, "index"])
                if lock_gt[threshold_idx, original_gt_idx] >= 0:
                    continue
                tp[threshold_idx, pred_idx] = 1
                lock_gt[threshold_idx, original_gt_idx] = pred_idx
                break
            if fp[threshold_idx, pred_idx] == 0 and tp[threshold_idx, pred_idx] == 0:
                fp[threshold_idx, pred_idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1)
    fp_cumsum = np.cumsum(fp, axis=1)
    recall = tp_cumsum / npos
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
    for threshold_idx in range(len(tiou_thresholds)):
        ap[threshold_idx] = interpolated_prec_rec(precision[threshold_idx, :], recall[threshold_idx, :])
    return ap


def temporal_map_from_eval(eval_df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    ground_truth = _runs_as_segments(eval_df, "y_true")
    prediction = _runs_as_segments(eval_df, "y_pred")
    ap = np.zeros((len(TIOU_THRESHOLDS), len(ACTIVE_LABELS)), dtype=float)
    for label_index, label in enumerate(ACTIVE_LABELS):
        ap[:, label_index] = average_precision_for_class(
            ground_truth[ground_truth["label_name"].eq(label)].reset_index(drop=True),
            prediction[prediction["label_name"].eq(label)].reset_index(drop=True),
            tiou_thresholds=TIOU_THRESHOLDS,
        )
    mAP_by_tiou = ap.mean(axis=1) if ap.size else np.zeros(len(TIOU_THRESHOLDS), dtype=float)
    detail = {f"mAP_tiou_{str(threshold).replace('.', '_')}": float(value) for threshold, value in zip(TIOU_THRESHOLDS, mAP_by_tiou)}
    return float(mAP_by_tiou.mean()), detail


def metric_row(eval_df: pd.DataFrame) -> dict[str, float | int]:
    y_true = eval_df["y_true"].astype(str).to_numpy()
    y_pred = eval_df["y_pred"].astype(str).to_numpy()
    active_mask = y_true != "Still"
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=1,
    )
    mAP, mAP_detail = temporal_map_from_eval(eval_df)
    row: dict[str, float | int] = {
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "sample_accuracy": float(accuracy_score(y_true, y_pred)),
        "active_accuracy": float(accuracy_score(y_true[active_mask], y_pred[active_mask])) if active_mask.any() else math.nan,
        "mAP": float(mAP),
        "n_samples": int(len(y_true)),
        "active_samples": int(active_mask.sum()),
    }
    row.update(mAP_detail)
    return row


def save_confusion(eval_df: pd.DataFrame, output_dir: str | Path, filename_prefix: str, title: str) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(eval_df["y_true"], eval_df["y_pred"], labels=LABEL_ORDER)
    counts = pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER)
    counts.index.name = "manual_label"
    csv_path = output_dir / f"{filename_prefix}_confusion_counts.csv"
    png_path = output_dir / f"{filename_prefix}_confusion_matrix.png"
    counts.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="white")
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABEL_ORDER)
    display.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return csv_path, png_path


def _prediction_path_for_run(prediction_root: Path, annotation_id: str, method_id: str, run_id: str) -> Path | None:
    run_dir = prediction_root / annotation_id / method_id / run_id
    preferred = run_dir / "data.json"
    if preferred.exists():
        return preferred
    if run_dir.exists():
        jsons = sorted(path for path in run_dir.glob("*.json") if path.is_file())
        if jsons:
            return jsons[0]
    return None


def _representative_run(run_rows: list[dict[str, object]]) -> int:
    if len(run_rows) <= 1:
        return 0
    f1 = pd.to_numeric(pd.Series([row.get("macro_f1") for row in run_rows]), errors="coerce")
    if f1.notna().any():
        return int((f1 - float(f1.mean())).abs().fillna(float("inf")).idxmin())
    return 0


def _write_repeated_reports(run_rows: list[dict[str, object]], figure_dir: Path, annotation_id: str, method_id: str) -> tuple[Path, Path, dict[str, object]]:
    method_dir = figure_dir / method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    run_df = pd.DataFrame(run_rows)
    run_csv = method_dir / f"{annotation_id}_{method_id}_repeated_run_metrics.csv"
    run_df.to_csv(run_csv, index=False)

    summary: dict[str, object] = {
        "method_id": method_id,
        "display_name": METHOD_DISPLAY_NAMES.get(method_id, method_id),
        "run_count": int(len(run_df)),
        "run_ids": ", ".join(run_df["run_id"].astype(str).tolist()),
    }
    for metric in REPORT_METRICS:
        values = pd.to_numeric(run_df[metric], errors="coerce")
        summary[metric] = float(values.mean()) if values.notna().any() else math.nan
        summary[f"{metric}_std"] = float(values.std(ddof=0)) if values.notna().any() else math.nan
        summary[f"{metric}_min"] = float(values.min()) if values.notna().any() else math.nan
        summary[f"{metric}_max"] = float(values.max()) if values.notna().any() else math.nan
    summary_csv = method_dir / f"{annotation_id}_{method_id}_repeated_run_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    return run_csv, summary_csv, summary


def _format_mean_std(row: pd.Series, metric: str) -> str:
    mean = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
    std = pd.to_numeric(pd.Series([row.get(f"{metric}_std")]), errors="coerce").iloc[0]
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.3f}"
    pm = chr(177)
    return f"{mean:.3f} {pm} {std:.3f}"


def _display_metrics_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary_df.to_dict(orient="records"):
        series = pd.Series(row)
        rows.append(
            {
                "Method": row["display_name"],
                "Runs": int(row.get("run_count", 0)),
                "P": _format_mean_std(series, "macro_precision"),
                "R": _format_mean_std(series, "macro_recall"),
                "F1": _format_mean_std(series, "macro_f1"),
                "Sample Acc": _format_mean_std(series, "sample_accuracy"),
                "Active Acc": _format_mean_std(series, "active_accuracy"),
                "mAP": _format_mean_std(series, "mAP"),
            }
        )
    return pd.DataFrame(rows)


def evaluate_llm_predictions(
    manual_label_csv: str | Path,
    prediction_root: str | Path,
    output_root: str | Path,
    *,
    annotation_id: str | None = None,
    window_index_csv: str | Path | None = None,
    method_ids: list[str] | None = None,
    n_runs: int = 4,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
) -> EvaluationResult:
    manual_df = load_manual_labels(manual_label_csv)
    if annotation_id is None:
        annotation_id = str(manual_df["annotation_id"].iloc[0])
    method_ids = method_ids or DEFAULT_METHODS
    prediction_root = Path(prediction_root)
    figure_dir = Path(output_root) / "figs" / annotation_id
    figure_dir.mkdir(parents=True, exist_ok=True)

    method_tables: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []

    for method_id in method_ids:
        run_rows: list[dict[str, object]] = []
        run_tables: list[tuple[str, pd.DataFrame]] = []
        for run_index in range(1, int(n_runs) + 1):
            run_id = f"run_{run_index:02d}"
            prediction_path = _prediction_path_for_run(prediction_root, annotation_id, method_id, run_id)
            if prediction_path is None:
                missing_rows.append(
                    {
                        "method_id": method_id,
                        "run_id": run_id,
                        "status": "missing",
                        "expected_dir": str(prediction_root / annotation_id / method_id / run_id),
                        "prediction_path": "",
                        "error": "",
                    }
                )
                continue

            method_display = METHOD_DISPLAY_NAMES.get(method_id, method_id)
            run_dir = figure_dir / method_id / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{annotation_id}_{method_id}_{run_id}"

            try:
                if "timeline" in method_id:
                    y_pred = rasterize_llm_timeline(prediction_path, manual_df, sampling_rate_hz=sampling_rate_hz)
                else:
                    if window_index_csv is None:
                        raise ValueError(f"window_index_csv is required for window-level method: {method_id}")
                    y_pred = rasterize_llm_windows(prediction_path, manual_df, window_index_csv, sampling_rate_hz=sampling_rate_hz)

                table = eval_table(annotation_id, manual_df, y_pred)
                metrics = metric_row(table)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError, OSError) as exc:
                missing_rows.append(
                    {
                        "method_id": method_id,
                        "run_id": run_id,
                        "status": "skipped_invalid",
                        "expected_dir": str(prediction_root / annotation_id / method_id / run_id),
                        "prediction_path": str(prediction_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            sample_csv = run_dir / f"{prefix}_sample_predictions.csv"
            table.to_csv(sample_csv, index=False)
            confusion_csv, confusion_png = save_confusion(table, run_dir, prefix, f"{annotation_id}: {method_display} {run_id}")
            timeline_png = save_timeline_plot(manual_df, table, annotation_id, f"{method_display} {run_id}", run_dir / f"{prefix}_timeline.png")

            run_row: dict[str, object] = {
                "method_id": method_id,
                "display_name": method_display,
                "run_id": run_id,
                "prediction_path": str(prediction_path),
                "sample_predictions_csv": str(sample_csv),
                "confusion_counts_csv": str(confusion_csv),
                "confusion_matrix_png": str(confusion_png),
                "timeline_png": str(timeline_png),
            }
            run_row.update(metrics)
            run_rows.append(run_row)
            run_tables.append((run_id, table))

        if not run_rows:
            continue

        repeated_timeline = save_repeated_run_timeline(
            manual_df,
            run_tables,
            annotation_id,
            method_id,
            METHOD_DISPLAY_NAMES.get(method_id, method_id),
            figure_dir,
        )
        run_csv, summary_csv, summary = _write_repeated_reports(run_rows, figure_dir, annotation_id, method_id)
        summary.update(
            {
                "run_metrics_csv": str(run_csv),
                "run_summary_csv": str(summary_csv),
                "repeated_timeline_png": str(repeated_timeline),
            }
        )
        summary_rows.append(summary)
        representative_index = _representative_run(run_rows)
        method_tables[method_id] = run_tables[representative_index][1]

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = figure_dir / f"{annotation_id}_model_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    table_csv = figure_dir / f"{annotation_id}_model_metrics_table.csv"
    table_png = figure_dir / f"{annotation_id}_model_metrics_table.png"
    if summary_df.empty:
        display_table = pd.DataFrame(
            [
                {
                    "Method": "No valid LLM predictions found yet",
                    "Runs": 0,
                    "P": "",
                    "R": "",
                    "F1": "",
                    "Sample Acc": "",
                    "Active Acc": "",
                    "mAP": "",
                }
            ]
        )
    else:
        display_table = _display_metrics_table(summary_df)
    display_table.to_csv(table_csv, index=False)
    render_metrics_table_png(display_table, table_png)

    combined_png = None
    if method_tables:
        combined_png = save_combined_timeline(manual_df, method_tables, METHOD_DISPLAY_NAMES, annotation_id, figure_dir)

    missing_csv = None
    if missing_rows:
        missing_csv = figure_dir / f"{annotation_id}_missing_predictions.csv"
        pd.DataFrame(missing_rows).to_csv(missing_csv, index=False)

    return EvaluationResult(
        figure_dir=figure_dir,
        model_metrics_csv=table_csv,
        model_metrics_png=table_png,
        combined_timeline_png=combined_png,
        missing_predictions_csv=missing_csv,
        summary=summary_df,
    )
