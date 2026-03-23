from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AudioParams:
    sample_rate: int = 32000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 320
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 16000.0
    window_seconds: float = 5.0
    window_stride_seconds: float = 5.0
    log_offset: float = 1e-5

    @property
    def window_num_samples(self) -> int:
        return int(round(self.window_seconds * self.sample_rate))

    @property
    def stride_num_samples(self) -> int:
        return int(round(self.window_stride_seconds * self.sample_rate))

    @property
    def window_num_frames(self) -> int:
        return int(round(self.window_seconds * self.sample_rate / self.hop_length))

    @property
    def stride_num_frames(self) -> int:
        return int(round(self.window_stride_seconds * self.sample_rate / self.hop_length))


def _hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray:
    freq = np.asarray(freq_hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(freq_mel: np.ndarray | float) -> np.ndarray:
    mel = np.asarray(freq_mel, dtype=np.float64)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(params: AudioParams) -> torch.Tensor:
    n_freqs = params.n_fft // 2 + 1
    mel_min = _hz_to_mel(params.fmin)
    mel_max = _hz_to_mel(params.fmax)
    mel_points = np.linspace(mel_min, mel_max, params.n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((params.n_fft + 1) * hz_points / params.sample_rate).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    filterbank = np.zeros((params.n_mels, n_freqs), dtype=np.float32)
    for mel_idx in range(params.n_mels):
        left = bins[mel_idx]
        center = bins[mel_idx + 1]
        right = bins[mel_idx + 2]
        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs - 1)
        if center > left:
            filterbank[mel_idx, left:center] = np.linspace(
                0.0, 1.0, center - left, endpoint=False, dtype=np.float32
            )
        if right > center:
            filterbank[mel_idx, center:right] = np.linspace(
                1.0, 0.0, right - center, endpoint=False, dtype=np.float32
            )
    return torch.tensor(filterbank, dtype=torch.float32)


def load_audio_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    audio, src_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if src_sr != sample_rate:
        gcd = math.gcd(src_sr, sample_rate)
        audio = resample_poly(audio, sample_rate // gcd, src_sr // gcd).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def compute_logmel(audio: np.ndarray | torch.Tensor, params: AudioParams) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        waveform = torch.tensor(audio, dtype=torch.float32)
    else:
        waveform = audio.to(dtype=torch.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape {tuple(waveform.shape)}")

    window = torch.hann_window(params.win_length, periodic=True)
    stft = torch.stft(
        waveform,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        win_length=params.win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    power = stft.abs().pow(2.0)
    mel_filter = build_mel_filterbank(params)
    mel = mel_filter @ power
    return torch.log(mel.clamp_min(params.log_offset))


def window_spectrogram(
    logmel: torch.Tensor,
    params: AudioParams,
    start_frame: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    num_frames = params.window_num_frames
    end_frame = start_frame + num_frames
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame <= logmel.shape[1]:
        window = logmel[:, start_frame:end_frame]
    else:
        pad_frames = end_frame - logmel.shape[1]
        window = F.pad(logmel[:, start_frame:], (0, pad_frames), value=pad_value)
    return window


def iter_window_frames(num_frames: int, params: AudioParams) -> Iterable[int]:
    if num_frames <= 0:
        yield 0
        return
    max_start = max(num_frames - params.window_num_frames, 0)
    starts = list(range(0, max_start + 1, max(params.stride_num_frames, 1)))
    if not starts or starts[-1] != max_start:
        starts.append(max_start)
    for start in starts:
        yield int(start)


def count_windows(num_frames: int, params: AudioParams) -> int:
    return sum(1 for _ in iter_window_frames(num_frames, params))

