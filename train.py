# training/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE contrastive loss for text-video retrieval.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, video_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # Similarity matrix [B, B]
        logits = (video_emb @ text_emb.t()) / self.temperature

        targets = torch.arange(logits.size(0), device=logits.device)

        loss_v2t = F.cross_entropy(logits, targets)
        loss_t2v = F.cross_entropy(logits.t(), targets)

        return 0.5 * (loss_v2t + loss_t2v)
