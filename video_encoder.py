# models/text_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    Input tokens dict:
      input_ids:      [B, L]
      attention_mask: [B, L]
    Output:
      text_emb:       [B, D]
    Strategy:
      - Transformer outputs last_hidden_state [B, L, H]
      - Mean pooling over valid tokens (attention_mask)
      - MLP projection -> [B, embed_dim]
      - optional L2 normalization
    """

    def __init__(self, text_model_name="distilbert-base-uncased", freeze=False,
                 embed_dim=256, dropout=0.1, normalize=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(text_model_name)
        hidden_dim = self.model.config.hidden_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.normalize = normalize

    def masked_mean_pool(self, last_hidden, attention_mask):
        # last_hidden: [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = (last_hidden * mask).sum(dim=1)     # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)      # [B, 1]
        return summed / denom

    def forward(self, tokens: dict) -> torch.Tensor:
        out = self.model(**tokens)
        last_hidden = out.last_hidden_state  # [B, L, H]
        pooled = self.masked_mean_pool(last_hidden, tokens["attention_mask"])  # [B, H]

        emb = self.proj(pooled)  # [B, D]
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb
