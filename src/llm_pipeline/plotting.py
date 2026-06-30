from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from .labels import LABEL_COLORS, LABEL_ORDER


def _use_light_style() -> None:
    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "white"
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"


def _style_axes(*axes) -> None:
    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelcolor="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for spine in ax.spines.values():
            spine.set_color("black")


def _style_legend(legend) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#bdbdbd")
    for text in legend.get_texts():
        text.set_color("black")


def sample_end_times(times: pd.Series) -> pd.Series:
    times = pd.to_datetime(times)
    next_times = times.shift(-1)
    diffs = times.diff().dropna()
    default_delta = diffs.median() if not diffs.empty else pd.Timedelta(milliseconds=20)
    if pd.isna(default_delta) or default_delta <= pd.Timedelta(0):
        default_delta = pd.Timedelta(milliseconds=20)
    next_times.iloc[-1] = times.iloc[-1] + default_delta
    return pd.to_datetime(next_times)


def _runs_from_intervals(frame: pd.DataFrame, label_col: str) -> list[tuple[float, float, str]]:
    if frame.empty:
        return []
    runs: list[tuple[float, float, str]] = []
    start_idx = 0
    labels = frame[label_col].astype(str).tolist()
    for index in range(1, len(frame) + 1):
        if index == len(frame) or labels[index] != labels[index - 1]:
            runs.append(
                (
                    float(frame.loc[start_idx, "rel_s"]),
                    float(frame.loc[index - 1, "end_rel_s"]),
                    labels[index - 1],
                )
            )
            start_idx = index
    return runs


def _sensor_with_relative_time(sensor_df: pd.DataFrame) -> pd.DataFrame:
    sensor = sensor_df.copy().sort_values("time").reset_index(drop=True)
    sensor["time"] = pd.to_datetime(sensor["time"])
    origin = sensor["time"].iloc[0]
    sensor["rel_s"] = (sensor["time"] - origin).dt.total_seconds()
    return sensor


def save_manual_overlay(manual_df: pd.DataFrame, output_path: str | Path, title: str | None = None) -> Path:
    _use_light_style()
    output_path = Path(output_path)
    sensor = _sensor_with_relative_time(manual_df)
    plot_df = sensor[["time", "manual_label"]].copy()
    plot_df["end_time"] = sample_end_times(plot_df["time"])
    plot_df["rel_s"] = sensor["rel_s"]
    plot_df["end_rel_s"] = (plot_df["end_time"] - sensor["time"].iloc[0]).dt.total_seconds()

    fig, ax = plt.subplots(figsize=(18, 5.5), facecolor="white")
    for start_s, end_s, label in _runs_from_intervals(plot_df, "manual_label"):
        ax.axvspan(start_s, end_s, color=LABEL_COLORS.get(label, "#cccccc"), alpha=0.24)
    for channel, color in [("x", "blue"), ("y", "red"), ("z", "black")]:
        ax.plot(sensor["rel_s"], sensor[channel], label=channel, color=color, linewidth=0.7)
    ax.set_title(title or "ACC with manual-label overlay")
    ax.set_xlabel("Relative time (s)")
    ax.set_ylabel("Acceleration")
    handles = [Patch(facecolor=LABEL_COLORS[label], edgecolor="gray", label=label) for label in LABEL_ORDER]
    line_handles, line_labels = ax.get_legend_handles_labels()
    _style_legend(ax.legend(line_handles + handles, line_labels + LABEL_ORDER, loc="upper right", ncol=4))
    _style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return output_path


def _plot_label_rows(
    ax,
    rows: list[tuple[str, list[tuple[float, float, str]]]],
    *,
    title: str,
    legend: bool = True,
) -> None:
    row_height = 1.0 / max(1, len(rows))
    for row_index, (_row_label, runs) in enumerate(rows):
        y0 = 1.0 - (row_index + 1) * row_height + 0.08 * row_height
        y1 = 1.0 - row_index * row_height - 0.08 * row_height
        for start_s, end_s, label in runs:
            ax.axvspan(start_s, end_s, ymin=y0, ymax=y1, color=LABEL_COLORS.get(label, "#cccccc"), alpha=0.55)

    tick_positions = [1.0 - (idx + 0.5) * row_height for idx in range(len(rows))]
    ax.set_ylim(0, 1)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("Relative time (s)")
    ax.set_title(title)
    if legend:
        handles = [Patch(facecolor=LABEL_COLORS[label], edgecolor="gray", label=label) for label in LABEL_ORDER]
        _style_legend(ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=5))


def _eval_runs(eval_df: pd.DataFrame, origin: pd.Timestamp, label_col: str) -> list[tuple[float, float, str]]:
    frame = eval_df.copy().reset_index(drop=True)
    frame["start_time"] = pd.to_datetime(frame["start_time"])
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    frame["rel_s"] = (frame["start_time"] - origin).dt.total_seconds()
    frame["end_rel_s"] = (frame["end_time"] - origin).dt.total_seconds()
    return _runs_from_intervals(frame, label_col)


def save_timeline_plot(
    sensor_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    annotation_id: str,
    model_name: str,
    output_path: str | Path,
) -> Path:
    _use_light_style()
    output_path = Path(output_path)
    sensor = _sensor_with_relative_time(sensor_df)
    origin = sensor["time"].iloc[0]

    rows = [
        ("Manual", _eval_runs(eval_df, origin, "y_true")),
        ("Prediction", _eval_runs(eval_df, origin, "y_pred")),
    ]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(18, 7.2),
        sharex=True,
        facecolor="white",
        gridspec_kw={"height_ratios": [3, 2]},
    )
    ax0, ax1 = axes
    for channel, color in [("x", "blue"), ("y", "red"), ("z", "black")]:
        ax0.plot(sensor["rel_s"], sensor[channel], label=channel, color=color, linewidth=0.7)
    ax0.set_title(f"{annotation_id}: {model_name}")
    ax0.set_ylabel("Acceleration")
    _style_legend(ax0.legend(loc="upper right", ncol=3))
    _plot_label_rows(ax1, rows, title="Manual labels and prediction")
    _style_axes(*axes)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return output_path


def save_repeated_run_timeline(
    sensor_df: pd.DataFrame,
    run_tables: list[tuple[str, pd.DataFrame]],
    annotation_id: str,
    method_id: str,
    method_display_name: str,
    output_dir: str | Path,
) -> Path:
    _use_light_style()
    output_dir = Path(output_dir)
    sensor = _sensor_with_relative_time(sensor_df)
    origin = sensor["time"].iloc[0]
    manual_eval = run_tables[0][1] if run_tables else pd.DataFrame()
    rows = [("Manual", _eval_runs(manual_eval, origin, "y_true"))]
    rows.extend((run_id, _eval_runs(table, origin, "y_pred")) for run_id, table in run_tables)

    fig_height = 3.2 + 0.68 * len(rows)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(18, fig_height),
        sharex=True,
        facecolor="white",
        gridspec_kw={"height_ratios": [3, max(2, len(rows))]},
    )
    ax0, ax1 = axes
    for channel, color in [("x", "blue"), ("y", "red"), ("z", "black")]:
        ax0.plot(sensor["rel_s"], sensor[channel], label=channel, color=color, linewidth=0.7)
    ax0.set_title(f"{annotation_id}: {method_display_name} repeated LLM runs")
    ax0.set_ylabel("Acceleration")
    _style_legend(ax0.legend(loc="upper right", ncol=3))
    _plot_label_rows(ax1, rows, title="Manual labels and independent LLM predictions")
    _style_axes(*axes)
    output_path = output_dir / method_id / f"{annotation_id}_{method_id}_timeline.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return output_path


def save_combined_timeline(
    sensor_df: pd.DataFrame,
    method_tables: dict[str, pd.DataFrame],
    display_names: dict[str, str],
    annotation_id: str,
    output_dir: str | Path,
) -> Path:
    _use_light_style()
    output_dir = Path(output_dir)
    sensor = _sensor_with_relative_time(sensor_df)
    origin = sensor["time"].iloc[0]
    first_table = next(iter(method_tables.values())) if method_tables else pd.DataFrame()
    rows = [("Manual", _eval_runs(first_table, origin, "y_true"))]
    rows.extend((display_names.get(method_id, method_id), _eval_runs(table, origin, "y_pred")) for method_id, table in method_tables.items())

    fig_height = 3.2 + 0.74 * len(rows)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(18, fig_height),
        sharex=True,
        facecolor="white",
        gridspec_kw={"height_ratios": [3, max(2, len(rows))]},
    )
    ax0, ax1 = axes
    for channel, color in [("x", "blue"), ("y", "red"), ("z", "black")]:
        ax0.plot(sensor["rel_s"], sensor[channel], label=channel, color=color, linewidth=0.7)
    ax0.set_title(f"{annotation_id}: LLM variant comparison")
    ax0.set_ylabel("Acceleration")
    _style_legend(ax0.legend(loc="upper right", ncol=3))
    _plot_label_rows(ax1, rows, title="Manual labels and representative LLM predictions")
    _style_axes(*axes)
    output_path = output_dir / f"{annotation_id}_llm_variants_combined_timeline.png"
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return output_path


def render_metrics_table_png(table: pd.DataFrame, output_path: str | Path) -> Path:
    _use_light_style()
    output_path = Path(output_path)
    display = table.copy()
    fig_height = max(2.2, 0.48 * (len(display) + 1.6))
    fig, ax = plt.subplots(figsize=(12.5, fig_height), facecolor="white")
    ax.axis("off")
    tbl = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for (row, _col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bdbdbd")
        if row == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return output_path
