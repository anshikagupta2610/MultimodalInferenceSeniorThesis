# GTA speech-only baseline — Vertex AI + P1 evaluation

This project has **four Python files and one reference JSON**. It uses the group's Google Cloud project, **not an API key**. The JSON is the corrected viewer-facing Method 1 proxy reference confirmed in the original gallery.

## One-time local setup

```bash
python -m pip install -r requirements.txt
gcloud config set project cos-multimodalinputs
gcloud auth application-default login
# Optional if you get quota-project errors (requires Service Usage Consumer permission):
gcloud auth application-default set-quota-project cos-multimodalinputs
cp .env.example .env
```

The `.env` needs:

```dotenv
GOOGLE_CLOUD_PROJECT=cos-multimodalinputs
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=true
```

**No `GEMINI_API_KEY` or `GOOGLE_API_KEY` is needed.** Your own machine must have working Google Cloud ADC and authorization to call Vertex AI. Model defaults to `gemini-2.5-flash` (override with `--model`). Vertex usage may incur charges to the group project. Neither this package nor ChatGPT contains your team credentials or actual participant recordings.

## Original data

Your original study data must be available locally: `Data/P1/1/1/transcript.csv`, `Data/P1/tower_list.txt`, etc. Pass the directory **containing P1** as `--data-root`. `dataset.py` reads the transcript and reserves the tower ID **locally** for evaluation; Gemini sees only speech.

## Test first: P1 Trial 1 (full trial, all step messages)

```bash
python evaluate.py batch --data-root '/absolute/path/to/Data' --participant P1 --trial 1 --output-root outputs/p1_trial1
```

This sends ONE structured-output Gemini request, saves `prediction.json`, `tower.json`, `tower.png`, `dataset_input.json` in the trial's `runs/` folder (if buildable), then writes:

- `outputs/p1_trial1/per_trial.csv` — one row per trial
- `outputs/p1_trial1/summary.json` — aggregate metrics and denominators
- `outputs/p1_trial1/results.json` — predicted/ground-truth blocks, matches, skipped failures

Use `--dry-run` to list selections with **no cloud calls**. If Gemini reports ambiguity, that trial is counted as no completed tower rather than silently removed. API failures are logged separately and not counted as incorrect guesses.

## Next: all 12 P1 trials

```bash
python evaluate.py batch --data-root '/absolute/path/to/Data' --participant P1 --output-root outputs/p1_all
```

By default, this uses **full-trial text**, concatenating every step message for each trial into one request. Set `--selection first-step` to evaluate only the first instruction of each trial instead. You should compare like with like across trials. `--limit 1` runs a pilot; `--resume` re-evaluates saved results only when transcript/model/provenance match, without issuing new requests.

## Optional: incremental *live* replay (one recorded step)

```bash
python live.py replay --data-root '/absolute/path/to/Data' --participant P1 --trial 1 --step 1 --output outputs/p1_live_t1_s1
python evaluate.py proxy --run outputs/p1_live_t1_s1 --output-dir outputs/p1_live_accuracy
python live.py video --log outputs/p1_live_t1_s1/live_log.json --output outputs/p1_live_t1_s1/reconstruction.mp4
```

This uses the time-stamped transcript and **multiple requests only as speech becomes available**. It is a *different experimental condition* from the single request full-trial baseline and should not be pooled with it. It is a visualization of predicted blocks, not Gemini-generated video. MP4 requires `ffmpeg`. `--headless --no-audio` can be used if desired.

## Visual comparison and tests

```bash
python evaluate.py plot --report outputs/p1_trial1/results.json --index 0 --output outputs/p1_trial1/comparison.png
python evaluate.py test
```

## Interpreting scores

Scores are on the corrected **simplified** target representation, not exact scanned Builder constructions. `full_tower_exact` = all three blocks matching in color, 3D center and orientation within a **fixed 0.25-V2-unit tolerance** (no extra blocks). `color_f1`, `xz_f1`, `xyz_f1`, `oriented_f1` progressively demand more detail. A prediction can be the right shape but wrong exact arrangement. F1 and exact scores are averaged over *scored* trials; failed Vertex requests are reported separately. Check sample images, especially frame orientation, before publishing results.

Do not add `--flip-z`: the JSON already contains the viewer-Z correction. The old mesh, raw transform, and CSV files are not needed in this active folder; archive them separately as source provenance.
