from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(audio_path: str | Path, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)

    if y.size == 0:
        y = np.zeros(sample_rate, dtype=np.float32)
    return y, sr


def summarize_matrix(features: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            features.mean(axis=1),
            features.std(axis=1),
            features.min(axis=1),
            features.max(axis=1),
        ]
    )


def extract_mfcc_stats_from_signal(y: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    """Return fixed-length MFCC statistics for one audio signal."""

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    return summarize_matrix(features)


def extract_mfcc_stats(audio_path: str | Path, sample_rate: int = 16000, n_mfcc: int = 40) -> np.ndarray:
    y, sr = load_audio(audio_path, sample_rate=sample_rate)
    return extract_mfcc_stats_from_signal(y, sr, n_mfcc=n_mfcc)


def build_audio_feature_matrix(audio_paths: list[str], sample_rate: int = 16000) -> np.ndarray:
    return np.vstack([extract_mfcc_stats(path, sample_rate=sample_rate) for path in audio_paths])


def summarize_track(values: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array([values.mean(), values.std(), values.min(), values.max()], dtype=np.float32)


def extract_enhanced_stats_from_signal(
    y: np.ndarray, sr: int, n_mfcc: int = 40, include_pitch: bool = True
) -> np.ndarray:
    """MFCCs plus chroma, energy, pitch, and spectral summary features."""
    mfcc_stats = extract_mfcc_stats_from_signal(y, sr, n_mfcc=n_mfcc)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    pitch_stats = np.zeros(4, dtype=np.float32)
    if include_pitch:
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        pitch_stats = summarize_track(f0)

    prosody = np.concatenate(
        [
            summarize_track(rms),
            summarize_track(zcr),
            summarize_track(centroid),
            summarize_track(bandwidth),
            summarize_track(rolloff),
            summarize_track(flatness),
            summarize_matrix(chroma),
            summarize_matrix(contrast),
            pitch_stats,
        ]
    )
    return np.concatenate([mfcc_stats, prosody])


def extract_mfcc_prosody_stats(
    audio_path: str | Path, sample_rate: int = 16000, n_mfcc: int = 40
) -> np.ndarray:
    y, sr = load_audio(audio_path, sample_rate=sample_rate)
    return extract_enhanced_stats_from_signal(y, sr, n_mfcc=n_mfcc, include_pitch=False)


def extract_enhanced_stats(
    audio_path: str | Path,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    include_pitch: bool = False,
) -> np.ndarray:
    y, sr = load_audio(audio_path, sample_rate=sample_rate)
    return extract_enhanced_stats_from_signal(y, sr, n_mfcc=n_mfcc, include_pitch=include_pitch)


def build_mfcc_prosody_feature_matrix(
    audio_paths: list[str], sample_rate: int = 16000
) -> np.ndarray:
    return np.vstack(
        [extract_mfcc_prosody_stats(path, sample_rate=sample_rate) for path in audio_paths]
    )


def build_enhanced_feature_matrix(
    audio_paths: list[str], sample_rate: int = 16000, include_pitch: bool = False
) -> np.ndarray:
    return np.vstack(
        [
            extract_enhanced_stats(path, sample_rate=sample_rate, include_pitch=include_pitch)
            for path in audio_paths
        ]
    )


def augment_signal(y: np.ndarray, sr: int) -> list[np.ndarray]:
    noise = np.random.default_rng(42).normal(0, 0.005 * max(np.std(y), 1e-4), size=y.shape)
    noisy = y + noise
    louder = np.clip(y * 1.15, -1.0, 1.0)
    quieter = y * 0.85
    pitch_up = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=1.0)
    stretched = librosa.effects.time_stretch(y, rate=1.05)
    return [noisy.astype(np.float32), louder.astype(np.float32), quieter.astype(np.float32), pitch_up, stretched]


def build_augmented_enhanced_feature_matrix(
    audio_paths: list[str], labels: list[str], sample_rate: int = 16000
) -> tuple[np.ndarray, list[str]]:
    features = []
    augmented_labels = []
    for path, label in zip(audio_paths, labels):
        y, sr = load_audio(path, sample_rate=sample_rate)
        signals = [y] + augment_signal(y, sr)
        for signal in signals:
            features.append(extract_enhanced_stats_from_signal(signal, sr, include_pitch=False))
            augmented_labels.append(label)
    return np.vstack(features), augmented_labels


def build_feature_matrix(
    audio_paths: list[str], sample_rate: int = 16000, feature_set: str = "mfcc"
) -> np.ndarray:
    if feature_set == "enhanced_pitch":
        return build_enhanced_feature_matrix(audio_paths, sample_rate=sample_rate, include_pitch=True)
    if feature_set == "enhanced":
        return build_enhanced_feature_matrix(audio_paths, sample_rate=sample_rate, include_pitch=False)
    if feature_set == "mfcc_prosody":
        return build_mfcc_prosody_feature_matrix(audio_paths, sample_rate=sample_rate)
    return build_audio_feature_matrix(audio_paths, sample_rate=sample_rate)
