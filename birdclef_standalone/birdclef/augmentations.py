from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class AugmentationConfig:
    mixup_alpha: float = 0.4
    gain_db: float = 6.0
    eq_strength: float = 0.2
    time_shift_frames: int = 40
    time_mask_width: int = 48
    freq_mask_width: int = 12
    noise_snr_db: float = 12.0


def _randint(max_value: int) -> int:
    if max_value <= 0:
        return 0
    return random.randint(0, max_value)


class SpectrogramAugmenter:
    def __init__(self, config: AugmentationConfig | None = None):
        self.config = config or AugmentationConfig()

    def apply(self, spectrogram: torch.Tensor, noise_bank: list[torch.Tensor] | None = None) -> torch.Tensor:
        x = spectrogram.clone()
        x = self.random_gain(x)
        x = self.random_eq(x)
        x = self.random_time_shift(x)
        x = self.random_freq_mask(x)
        x = self.random_time_mask(x)
        if noise_bank:
            x = self.random_background_mix(x, noise_bank)
        return x

    def random_gain(self, x: torch.Tensor) -> torch.Tensor:
        gain_db = random.uniform(-self.config.gain_db, self.config.gain_db)
        return x + gain_db / 20.0

    def random_eq(self, x: torch.Tensor) -> torch.Tensor:
        n_mels = x.shape[-2]
        anchor = torch.rand(4, device=x.device) * 2.0 - 1.0
        curve = torch.nn.functional.interpolate(
            anchor.view(1, 1, -1),
            size=n_mels,
            mode="linear",
            align_corners=True,
        ).view(n_mels, 1)
        return x + curve * self.config.eq_strength

    def random_time_shift(self, x: torch.Tensor) -> torch.Tensor:
        shift = random.randint(-self.config.time_shift_frames, self.config.time_shift_frames)
        return torch.roll(x, shifts=shift, dims=-1)

    def random_time_mask(self, x: torch.Tensor) -> torch.Tensor:
        width = _randint(self.config.time_mask_width)
        if width == 0 or width >= x.shape[-1]:
            return x
        start = _randint(x.shape[-1] - width)
        x[..., start : start + width] = x.mean()
        return x

    def random_freq_mask(self, x: torch.Tensor) -> torch.Tensor:
        width = _randint(self.config.freq_mask_width)
        if width == 0 or width >= x.shape[-2]:
            return x
        start = _randint(x.shape[-2] - width)
        x[..., start : start + width, :] = x.mean()
        return x

    def random_background_mix(self, x: torch.Tensor, noise_bank: list[torch.Tensor]) -> torch.Tensor:
        noise = random.choice(noise_bank)
        if noise.shape != x.shape:
            noise = torch.nn.functional.interpolate(
                noise.unsqueeze(0),
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        signal_power = x.pow(2).mean().clamp_min(1e-6)
        noise_power = noise.pow(2).mean().clamp_min(1e-6)
        snr_db = random.uniform(0.0, self.config.noise_snr_db)
        scale = torch.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10.0))))
        return x + scale * noise


def apply_mixup(
    inputs: torch.Tensor,
    hard_targets: torch.Tensor,
    soft_targets: torch.Tensor | None = None,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if alpha <= 0.0 or inputs.shape[0] < 2:
        return inputs, hard_targets, soft_targets
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(inputs.shape[0], device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    mixed_hard = lam * hard_targets + (1.0 - lam) * hard_targets[index]
    mixed_soft = None
    if soft_targets is not None:
        mixed_soft = lam * soft_targets + (1.0 - lam) * soft_targets[index]
    return mixed_inputs, mixed_hard, mixed_soft

