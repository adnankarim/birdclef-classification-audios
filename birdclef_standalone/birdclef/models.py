from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from birdclef.losses import linear_softmax_pooling


def _adapt_input_conv_to_single_channel(conv: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class EfficientNetV2SClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, use_mil: bool = False, dropout: float = 0.2):
        super().__init__()
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_v2_s(weights=weights)
        backbone.features[0][0] = _adapt_input_conv_to_single_channel(backbone.features[0][0])
        self.features = backbone.features
        self.conv_head = backbone.classifier[0]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1280, num_classes)
        self.use_mil = use_mil
        if use_mil:
            self.framewise_head = nn.Conv1d(1280, num_classes, kernel_size=1)
        else:
            self.framewise_head = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        clip_logits = self.classifier(self.dropout(pooled))
        outputs = {"clip_logits": clip_logits}
        if self.framewise_head is not None:
            time_features = features.mean(dim=2)
            frame_logits = self.framewise_head(time_features).transpose(1, 2)
            outputs["frame_logits"] = frame_logits
            outputs["pooled_logits"] = linear_softmax_pooling(frame_logits)
        return outputs


class PerchEmbeddingExtractor:
    """Optional offline extractor using bioacoustics-model-zoo if available.

    This is intentionally isolated from the main spectrogram pipeline because
    Perch is distributed through separate tooling. If the dependency is absent,
    callers get a clear error instead of a silent fallback.
    """

    def __init__(self):
        try:
            import bioacoustics_model_zoo as bmz  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Perch extraction requires `bioacoustics-model-zoo` or another "
                "user-supplied Perch embedding cache. Official Perch tooling is "
                "published via google-research/perch and google-research/perch-hoplite."
            ) from exc
        self._model = bmz.Perch2()

    def embed_windows(self, waveforms: list[np.ndarray], sample_rate: int) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="perch_embed_") as temp_dir:
            paths: list[str] = []
            for idx, waveform in enumerate(waveforms):
                temp_path = Path(temp_dir) / f"window_{idx}.wav"
                sf.write(temp_path, waveform, sample_rate)
                paths.append(str(temp_path))
            embeddings = self._model.embed(paths)
        if hasattr(embeddings, "to_numpy"):
            return embeddings.to_numpy(dtype=np.float32)
        if hasattr(embeddings, "values"):
            return np.asarray(embeddings.values, dtype=np.float32)
        return np.asarray(embeddings, dtype=np.float32)


class PerchMLPTeacher(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, hidden_dim: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"clip_logits": self.net(embeddings)}


class StudentExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        logits = outputs.get("pooled_logits", outputs["clip_logits"])
        return torch.sigmoid(logits)

