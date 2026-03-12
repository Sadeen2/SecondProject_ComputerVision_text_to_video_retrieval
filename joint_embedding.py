# models/config.py
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # ---- data ----
    num_frames: int = 8
    image_size: int = 224
    max_text_len: int = 32

    # ---- encoders ----
    backbone_name: str = "resnet50"   # resnet18/resnet34/resnet50
    pretrained: bool = True
    freeze_backbone: bool = False     # start False (fine-tune), can set True for faster training

    text_model_name: str = "distilbert-base-uncased"  # strong + lightweight
    freeze_text: bool = False

    # ---- embedding space ----
    embed_dim: int = 256              # shared embedding dimension for retrieval
    proj_dropout: float = 0.1
    normalize: bool = True            # L2 normalize for cosine similarity

    # ---- misc ----
    device: str = "cpu"
