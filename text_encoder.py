# models/joint_embedding.py
import torch
import torch.nn as nn

from .video_encoder import VideoEncoder
from .text_encoder import TextEncoder


class DualEncoderModel(nn.Module):
    """
    Dual-encoder model:
      - video encoder: CNN backbone + temporal pooling + projection
      - text encoder : Transformer + pooling + projection

    Returns embeddings in the same space for contrastive learning / retrieval.
    """

    def __init__(self, cfg):
        super().__init__()

        self.video_encoder = VideoEncoder(
            backbone_name=cfg.backbone_name,
            pretrained=cfg.pretrained,
            freeze_backbone=cfg.freeze_backbone,
            embed_dim=cfg.embed_dim,
            dropout=cfg.proj_dropout,
            normalize=cfg.normalize
        )

        self.text_encoder = TextEncoder(
            text_model_name=cfg.text_model_name,
            freeze=cfg.freeze_text,
            embed_dim=cfg.embed_dim,
            dropout=cfg.proj_dropout,
            normalize=cfg.normalize
        )

    def forward(self, video_frames: torch.Tensor, text_tokens: dict):
        video_emb = self.video_encoder(video_frames)
        text_emb = self.text_encoder(text_tokens)
        return video_emb, text_emb
