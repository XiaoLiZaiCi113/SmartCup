# Minimal COVFEE LLM Pipeline

This is the compact, reproducible version of the smart-cup LLM experiment pipeline.

The workflow is intentionally notebook-driven but keeps the running logic in Python modules:

1. Put the original COVFEE annotation JSON, the matching Midge ACC CSV file(s), and the video file on the local machine.
2. Open `notebooks/run_llm_pipeline.ipynb`.
3. Fill the first cell with absolute paths and alignment values:
   - `ANNOTATION_ID`, for example `01_v1`
   - `PARTICIPANT_NO`, for example `1`
   - `MIDGE_ID`, for example `81`
   - `ANNOTATION_JSON`, named like `<id>_<videoname>.json`
   - `ACC_PATHS`, one or more raw ACC CSV files
   - `VIDEO_PATH`, optional but recommended for traceability
   - `SOURCE_VIDEO_START_ABS_TIME`
   - `CLIP_START_OFFSET_S`
   - `CLIP_END_OFFSET_S`
4. Run the preprocessing and prompt cells.
5. Manually send each generated `prompt.txt` plus `input.json` to the LLM.
6. Save each LLM response as `data.json` in the prepared run folders.
7. Run the evaluation cell to produce per-run timelines, repeated-run summaries, `mean ± std` tables, and the combined representative timeline.

## Why Alignment Is Manual

The COVFEE annotation times are treated as clip-relative seconds. If a video was clipped from a source video, the video metadata may still refer to the source video rather than the clip. For that reason, the notebook explicitly asks for:

- source video absolute start time
- clip start offset in the source video
- clip end offset in the source video

The effective clip start is:

```text
VIDEO_START_ABS_TIME = SOURCE_VIDEO_START_ABS_TIME + CLIP_START_OFFSET_S
```

## Project Layout

```text
llm_pipeline/
  notebooks/
    run_llm_pipeline.ipynb
  src/
    llm_pipeline/
      preprocessing.py   # ACC loading, resampling, COVFEE annotation alignment
      prompts.py         # zero-shot/few-shot prompt and input JSON generation
      evaluation.py      # LLM output parsing, rasterization, metrics, repeated runs
      plotting.py        # timeline plots and metrics table rendering
      labels.py          # label maps, colors, aliases
  requirements.txt
```

## Generated Output Layout

```text
outputs/
  manual_labels/
  prompts/<annotation_id>/<method>/prompt.txt
  input_json/<annotation_id>/<method>/input.json
  input_json/<annotation_id>/window_index_<annotation_id>.csv
  llm_predictions/<annotation_id>/<method>/run_01/data.json
  llm_predictions/<annotation_id>/<method>/run_02/data.json
  llm_predictions/<annotation_id>/<method>/run_03/data.json
  llm_predictions/<annotation_id>/<method>/run_04/data.json
  figs/<annotation_id>/
```

Method names:

- `llm_zero_shot_xyz_timeline`
- `llm_few_shot_xyz_timeline`
- `llm_zero_shot_xyz_rag_tilt`
- `llm_few_shot_xyz_rag_tilt`

The evaluator records each LLM run separately and then reports method-level averages as `mean ± std`.
Incomplete experiments are allowed: missing or invalid run files are skipped and written to `figs/<annotation_id>/<annotation_id>_missing_predictions.csv`.
