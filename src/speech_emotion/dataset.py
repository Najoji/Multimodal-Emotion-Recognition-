from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


KNOWN_EMOTIONS = {
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "pleasant surprise",
    "ps",
    "sad",
}

EMOTION_ALIASES = {
    "pleasant surprise": "pleasant_surprise",
    "ps": "pleasant_surprise",
}


@dataclass(frozen=True)
class TessItem:
    audio_path: Path
    transcript: str
    emotion: str


def normalize_emotion(value: str) -> str:
    value = value.strip().lower().replace("-", "_")
    value = value.replace("pleasant_surprise", "pleasant surprise")
    return EMOTION_ALIASES.get(value, value.replace(" ", "_"))


def parse_tess_file(path: Path) -> TessItem:
    """Parse transcript and emotion from a TESS wav filename/folder.

    Typical TESS filenames look like:
    OAF_back_angry.wav
    YAF_wire_ps.wav
    """
    stem = path.stem
    parts = stem.split("_")

    emotion = None
    transcript_parts: list[str] = []

    if len(parts) >= 3:
        candidate = normalize_emotion("_".join(parts[2:]))
        if candidate in {normalize_emotion(x) for x in KNOWN_EMOTIONS}:
            emotion = candidate
            transcript_parts = parts[1:2]

    if emotion is None:
        for parent in path.parents:
            candidate = normalize_emotion(parent.name.replace("OAF_", "").replace("YAF_", ""))
            if candidate in {normalize_emotion(x) for x in KNOWN_EMOTIONS}:
                emotion = candidate
                break

    if emotion is None:
        raise ValueError(f"Could not infer emotion label from {path}")

    if not transcript_parts:
        transcript_parts = parts[1:-1] if len(parts) > 2 else [stem]

    transcript = " ".join(transcript_parts).replace("_", " ").strip().lower()
    return TessItem(audio_path=path, transcript=transcript, emotion=emotion)


def load_tess_dataframe(data_dir: str | Path) -> pd.DataFrame:
    data_path = Path(data_dir)
    wav_files = sorted(data_path.rglob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found under {data_path}. Download TESS and extract it into data/tess."
        )

    unique_wav_files = {}
    for wav_path in wav_files:
        unique_wav_files.setdefault(wav_path.name.lower(), wav_path)

    rows = []
    skipped = []
    for wav_path in unique_wav_files.values():
        try:
            item = parse_tess_file(wav_path)
            rows.append(
                {
                    "audio_path": str(item.audio_path),
                    "transcript": item.transcript,
                    "emotion": item.emotion,
                }
            )
        except ValueError as exc:
            skipped.append(str(exc))

    if not rows:
        sample = "\n".join(skipped[:5])
        raise ValueError(f"Could not parse any TESS files. First skipped files:\n{sample}")

    return pd.DataFrame(rows)
