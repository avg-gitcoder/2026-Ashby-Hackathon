"""Five independent morphology predictions from pretrained MUSK features."""
import torch
from torch import nn
from torch.nn import functional as F

FEATURES = (
    "abnormal_gland_arch", "nuclear_atypia", "abnormal_nuclear_org",
    "stromal_invasion", "intraluminal_necrosis",
)


class MorphologyModel(nn.Module):
    def __init__(self, encoder, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        # Single-scale MUSK CLS features have width 1024.
        self.head = nn.Linear(encoder.vision_head.in_features, 5 * 3)
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, images):
        with torch.set_grad_enabled(self.training and not self.freeze_encoder):
            features, _ = self.encoder(
                image=images, with_head=False, out_norm=False,
                ms_aug=False, return_global=True,
            )
        return self.head(features.to(self.head.weight.dtype)).reshape(-1, 5, 3)

    @torch.no_grad()
    def predict(self, images):
        if self.training:
            raise RuntimeError("Call model.eval() before predict().")
        probabilities = self(images).softmax(dim=-1)
        confidence, indices = probabilities.max(dim=-1)
        return {
            "labels": indices - 1,  # indices 0,1,2 -> labels -1,0,1
            "confidence": confidence,
            "probabilities": probabilities,
        }


def morphology_loss(logits, labels):
    """labels: [batch, 5], integer values -1, 0, 1; 0 is a real class."""
    if logits.ndim != 3 or logits.shape[1:] != (5, 3):
        raise ValueError("Expected logits shaped [batch, 5, 3].")
    if labels.shape != logits.shape[:2]:
        raise ValueError("Expected labels shaped [batch, 5].")
    if not torch.all((labels == -1) | (labels == 0) | (labels == 1)):
        raise ValueError("Labels must contain only -1, 0, 1.")
    return F.cross_entropy(logits.reshape(-1, 3), (labels + 1).long().reshape(-1))


def load_pretrained(freeze_encoder=True, checkpoint="hf_hub:xiangjx/musk"):
    from musk import modeling, utils  # registers MUSK with timm
    from timm.models import create_model

    encoder = create_model("musk_large_patch16_384")
    utils.load_model_and_may_interpolate(
        checkpoint, encoder, "model|module", ""
    )
    return MorphologyModel(encoder, freeze_encoder=freeze_encoder)
