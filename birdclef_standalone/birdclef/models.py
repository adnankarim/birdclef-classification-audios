from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torchvision import models

from birdclef.losses import linear_softmax_pooling


IMAGE_BACKBONE_NAMES = (
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "convnext_tiny",
    "convnext_small",
)
IMAGE_MODEL_TYPES = IMAGE_BACKBONE_NAMES + tuple(f"{name}_student" for name in IMAGE_BACKBONE_NAMES)


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


def canonical_image_model_type(model_type: str) -> str:
    if model_type.endswith("_student"):
        return model_type[: -len("_student")]
    return model_type


def is_supported_image_model_type(model_type: str) -> bool:
    return canonical_image_model_type(model_type) in IMAGE_BACKBONE_NAMES


def student_model_type(backbone: str) -> str:
    canonical = canonical_image_model_type(backbone)
    if canonical not in IMAGE_BACKBONE_NAMES:
        raise ValueError(f"Unsupported backbone={backbone!r}. Choices: {', '.join(IMAGE_BACKBONE_NAMES)}")
    return f"{canonical}_student"


class TorchvisionSpectrogramClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        use_mil: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        backbone_name = canonical_image_model_type(backbone_name)
        if backbone_name == "efficientnet_v2_s":
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_v2_s(weights=weights)
            feature_dim = 1280
        elif backbone_name == "efficientnet_v2_m":
            weights = models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_v2_m(weights=weights)
            feature_dim = 1280
        elif backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            feature_dim = 768
        elif backbone_name == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_small(weights=weights)
            feature_dim = 768
        else:
            raise ValueError(f"Unsupported backbone_name={backbone_name!r}. Choices: {', '.join(IMAGE_BACKBONE_NAMES)}")

        backbone.features[0][0] = _adapt_input_conv_to_single_channel(backbone.features[0][0])
        self.backbone_name = backbone_name
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.use_mil = use_mil
        if use_mil:
            self.framewise_head = nn.Conv1d(feature_dim, num_classes, kernel_size=1)
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


class EfficientNetV2SClassifier(TorchvisionSpectrogramClassifier):
    def __init__(self, num_classes: int, pretrained: bool = True, use_mil: bool = False, dropout: float = 0.2):
        super().__init__(
            backbone_name="efficientnet_v2_s",
            num_classes=num_classes,
            pretrained=pretrained,
            use_mil=use_mil,
            dropout=dropout,
        )


class EfficientNetV2MClassifier(TorchvisionSpectrogramClassifier):
    def __init__(self, num_classes: int, pretrained: bool = True, use_mil: bool = False, dropout: float = 0.2):
        super().__init__(
            backbone_name="efficientnet_v2_m",
            num_classes=num_classes,
            pretrained=pretrained,
            use_mil=use_mil,
            dropout=dropout,
        )


class ConvNeXtTinyClassifier(TorchvisionSpectrogramClassifier):
    def __init__(self, num_classes: int, pretrained: bool = True, use_mil: bool = False, dropout: float = 0.2):
        super().__init__(
            backbone_name="convnext_tiny",
            num_classes=num_classes,
            pretrained=pretrained,
            use_mil=use_mil,
            dropout=dropout,
        )


class ConvNeXtSmallClassifier(TorchvisionSpectrogramClassifier):
    def __init__(self, num_classes: int, pretrained: bool = True, use_mil: bool = False, dropout: float = 0.2):
        super().__init__(
            backbone_name="convnext_small",
            num_classes=num_classes,
            pretrained=pretrained,
            use_mil=use_mil,
            dropout=dropout,
        )


def build_image_classifier(
    model_type: str,
    num_classes: int,
    pretrained: bool = True,
    use_mil: bool = False,
    dropout: float = 0.2,
) -> TorchvisionSpectrogramClassifier:
    canonical = canonical_image_model_type(model_type)
    if canonical == "efficientnet_v2_s":
        return EfficientNetV2SClassifier(num_classes=num_classes, pretrained=pretrained, use_mil=use_mil, dropout=dropout)
    if canonical == "efficientnet_v2_m":
        return EfficientNetV2MClassifier(num_classes=num_classes, pretrained=pretrained, use_mil=use_mil, dropout=dropout)
    if canonical == "convnext_tiny":
        return ConvNeXtTinyClassifier(num_classes=num_classes, pretrained=pretrained, use_mil=use_mil, dropout=dropout)
    if canonical == "convnext_small":
        return ConvNeXtSmallClassifier(num_classes=num_classes, pretrained=pretrained, use_mil=use_mil, dropout=dropout)
    raise ValueError(f"Unsupported model_type={model_type!r}. Choices: {', '.join(IMAGE_MODEL_TYPES)}")


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


class PerchTemporalStudent(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.2,
        max_positions: int = 32,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.position_embed = nn.Embedding(max_positions, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.framewise_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if embeddings.ndim != 3:
            raise ValueError(f"Expected [batch, time, dim] embeddings, got {tuple(embeddings.shape)}")
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)
        x = self.input_proj(embeddings) + self.position_embed(positions)[None, :, :]
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.bool()
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        frame_logits = self.framewise_head(self.dropout(encoded))
        outputs = {"frame_logits": frame_logits}
        pooled_logits = linear_softmax_pooling(frame_logits)
        outputs["pooled_logits"] = pooled_logits
        return outputs


class StudentExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        logits = outputs.get("pooled_logits", outputs["clip_logits"])
        return torch.sigmoid(logits)
