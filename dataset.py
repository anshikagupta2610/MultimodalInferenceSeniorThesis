"""Load participant transcripts and per-trial metadata (never pass targets to Gemini)."""
from __future__ import annotations
import csv
import json
import math
import re
from pathlib import Path

def participant_id(value: str) -> str:
    match = re.fullmatch(r"[Pp]?([0-9]+)", str(value).strip())
    if not match or int(match[1]) < 1:
        raise ValueError("Participant must look like P1, P21, or 1.")
    return f"P{int(match[1])}"


def numeric_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    result = sorted((p for p in path.iterdir() if p.is_dir() and
                     p.name.isdigit() and int(p.name) > 0), key=lambda p: int(p.name))
    numbers = [int(p.name) for p in result]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"Duplicate numeric directory IDs in {path} (e.g. 1 and 01).")
    return result


def numbered_dir(parent: Path, value: int, kind: str) -> Path:
    if type(value) is not int or value < 1:
        raise ValueError(f"{kind} must be a positive integer.")
    directories = numeric_dirs(parent)
    for path in directories:
        if int(path.name) == value:
            return path
    available = ", ".join(p.name for p in directories) or "none"
    raise FileNotFoundError(f"{kind} {value} not found in {parent}; available: {available}.")


def read_segments(path: Path, words: bool) -> list[dict]:
    field = "word" if words else "text"
    segments = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"start", "end", field}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: expected CSV columns start,end,{field}.")
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"{path.name}, row {line}: extra CSV fields; quote text containing commas.")
            if not any(str(v or "").strip() for v in row.values()):
                continue
            text = (row.get(field) or "").strip()
            if not text:
                raise ValueError(f"{path.name}, row {line}: missing {field}.")
            try:
                start, end = float(row["start"]), float(row["end"])
            except (TypeError, ValueError):
                raise ValueError(f"{path.name}, row {line}: invalid timestamp.") from None
            if not all(math.isfinite(v) for v in (start, end)) or start < 0 or end < start:
                raise ValueError(f"{path.name}, row {line}: invalid timestamp interval.")
            segments.append({"start": start, "end": end, field: text})
    return sorted(segments, key=lambda s: s["start"])


def load_transcript(step_dir: Path) -> dict:
    candidates = sorted(p for p in step_dir.iterdir() if p.is_file()
                        and p.suffix.lower() == ".csv" and "transcript" in p.name.lower())
    sentence_files = [p for p in candidates if "word" not in p.name.lower()]
    word_files = [p for p in candidates if "word" in p.name.lower()]
    if len(sentence_files) > 1 or len(word_files) > 1:
        raise ValueError(f"Multiple transcript files in {step_dir}; keep one sentence and one word CSV.")
    sentences = read_segments(sentence_files[0], False) if sentence_files else []
    words = read_segments(word_files[0], True) if word_files else []
    if not sentences and not words:
        raise ValueError(f"No nonempty transcript CSV found in {step_dir}.")
    # Never concatenate sentence and word versions of the same speech.
    selected, field = (sentences, "text") if sentences else (words, "word")
    return {
        "text": " ".join(s[field] for s in selected),
        "sentences": sentences, "words": words, "clock": "audio_seconds_per_step",
        "source": "sentence_csv" if sentences else "word_csv_fallback",
        "sentence_file": sentence_files[0].name if sentence_files else None,
        "word_file": word_files[0].name if word_files else None,
    }


def read_target_for_log(pair_dir: Path, trial: int, step_dirs: list[Path]) -> dict:
    """Canonical list wins over step metadata. This result is NEVER model input."""
    warnings, observed = [], []
    tower_id = None
    source = None
    list_path = pair_dir / "tower_list.txt"
    if list_path.is_file():
        # Preserve blank lines so trial offsets do not shift silently.
        lines = list_path.read_text(encoding="utf-8-sig").splitlines()
        if trial <= len(lines) and lines[trial - 1].strip():
            tower_id, source = lines[trial - 1].strip(), "tower_list.txt"
        else:
            warnings.append("No canonical tower_list.txt entry for this trial.")
    else:
        warnings.append("tower_list.txt absent; canonical target unavailable.")
    for step in step_dirs:
        for name in ("metadata.json", "tower_meta.json"):
            path = step / name
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(data, dict):
                    raise ValueError("metadata must be an object")
                value = data.get("tower_id")
                if value is not None:
                    observed.append({"step": int(step.name), "file": name, "tower_id": str(value)})
                    if tower_id is not None and str(value) != tower_id:
                        warnings.append(f"Step {step.name} {name}: tower_id differs from canonical list.")
                for key, expected in (("trial_id", trial), ("step_num", int(step.name))):
                    if key in data and str(data[key]) != str(expected):
                        warnings.append(f"Step {step.name} {name}: {key} mismatch.")
            except (OSError, ValueError) as exc:
                warnings.append(f"Step {step.name} {name}: {type(exc).__name__} reading metadata.")
    return {"tower_id": tower_id, "source": source,
            "observed_step_targets": observed, "warnings": warnings}


def load_selection(data_root: Path | str, participant: str, trial: int,
                   step: int = 1, through_step: int | None = None) -> dict:
    root = Path(data_root).expanduser().resolve()
    pair = participant_id(participant)
    pair_dir = root / pair
    trial_dir = numbered_dir(pair_dir, trial, "Trial")
    if through_step is None:
        chosen = [numbered_dir(trial_dir, step, "Step")]
        mode = "single_step"
    else:
        numbered_dir(trial_dir, through_step, "Step")
        chosen = [p for p in numeric_dirs(trial_dir) if int(p.name) <= through_step]
        if [int(p.name) for p in chosen] != list(range(1, through_step + 1)):
            raise ValueError("Cumulative selection needs every step from 1 through the requested step.")
        mode = "cumulative_through_step"
    steps = []
    for directory in chosen:
        steps.append({"step_num": int(directory.name),
                      "relative_directory": directory.relative_to(root).as_posix(),
                      "language": load_transcript(directory)})
    # Numbering identifies a record; it does not establish R1/R4 membership.
    record = {"schema": "gta-v2-language-input-v1", "participant": pair,
              "trial": trial, "selection_mode": mode, "steps": steps,
              "target_for_local_log_only": read_target_for_log(pair_dir, trial, chosen)}
    record["model_input"] = model_input(record)
    return record


def model_input(record: dict) -> str:
    steps = record["steps"]
    if record["selection_mode"] == "single_step":
        return steps[0]["language"]["text"]
    preamble = ("Reconstruct the final tower described by these successive speech messages "
                "from one trial. Later messages can correct earlier ones. "
                "A correction that depends on an unseen Builder state may remain ambiguous.\n\n")
    return preamble + "\n\n".join(
        f"Message {s['step_num']}: {s['language']['text']}" for s in steps)


def inventory(data_root: Path | str) -> list[dict]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    pairs = sorted((p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"P[1-9][0-9]*", p.name)),
                   key=lambda p: int(p.name[1:]))
    return [{"participant": p.name, "trial": int(t.name),
             "steps": [int(s.name) for s in numeric_dirs(t)]}
            for p in pairs for t in numeric_dirs(p)]
