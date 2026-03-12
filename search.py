# retrieval/build_video_index.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer  # not strictly needed here, but kept for completeness

from data.video_text_dataset import MSVDDataset, collate_fn
from models.config import ModelConfig
from models.joint_embedding import DualEncoderModel
from utils.io_utils import ensure_dir, save_json, save_numpy, load_checkpoint


def detect_backbone_from_ckpt(ckpt: Dict) -> str:
    # your checkpoints sometimes stored cfg or backbone directly
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        return ckpt["cfg"].get("backbone_name", "resnet18")
    return ckpt.get("backbone", "resnet18")


def detect_text_model_from_ckpt(ckpt: Dict) -> str:
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        return ckpt["cfg"].get("text_model_name", "distilbert-base-uncased")
    return "distilbert-base-uncased"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Step 5A: Build cached video index (encode videos once).")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to model_best.pt")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_frames", type=int, default=4)
    ap.add_argument("--index_dir", type=str, default="index")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--persistent_workers", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt = load_checkpoint(args.checkpoint, map_location=str(device))

    backbone = detect_backbone_from_ckpt(ckpt)
    text_model = detect_text_model_from_ckpt(ckpt)

    cfg = ModelConfig(
        backbone_name=backbone,
        text_model_name=text_model,
        num_frames=args.num_frames,
        device=str(device),
    )

    print(f"Backbone: {cfg.backbone_name} | Text: {cfg.text_model_name}")
    print(f"num_frames: {cfg.num_frames} | image_size: {cfg.image_size}")

    model = DualEncoderModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Build dataset/loader (we only need frames + ids)
    ds = MSVDDataset(split=args.split, num_frames=args.num_frames, image_size=cfg.image_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
    )

    # We must deduplicate videos by video_id (many captions per video)
    vid2embed: Dict[str, np.ndarray] = {}

    print("Encoding videos (unique video_id)...")
    for batch in dl:
        frames = batch["video_frames"].to(device)          # [B,T,3,H,W]
        video_ids: List[str] = batch["video_ids"]          # list[str]

        # encode video only (ignore text)
        # We can call model.video_encoder directly to save time.
        v_emb = model.video_encoder(frames)                # [B, D]
        v_emb = torch.nn.functional.normalize(v_emb, dim=1)
        v_emb = v_emb.detach().cpu().numpy()

        for i, vid in enumerate(video_ids):
            # keep the first embedding for each video_id (same frames for same video_id)
            if vid not in vid2embed:
                vid2embed[vid] = v_emb[i]

    video_ids_unique = sorted(list(vid2embed.keys()))
    video_embeds = np.stack([vid2embed[v] for v in video_ids_unique], axis=0)

    index_dir = ensure_dir(args.index_dir)
    meta = {
        "dataset": "MSVD",
        "split": args.split,
        "backbone": cfg.backbone_name,
        "text_model": cfg.text_model_name,
        "num_frames": args.num_frames,
        "embed_dim": int(video_embeds.shape[1]),
        "n_videos": int(video_embeds.shape[0]),
    }

    save_json(index_dir / "meta.json", meta)
    save_json(index_dir / "video_ids.json", video_ids_unique)
    save_numpy(index_dir / "video_embeddings.npy", video_embeds)

    print("\n✅ Index saved:")
    print(f"  {index_dir / 'meta.json'}")
    print(f"  {index_dir / 'video_ids.json'}")
    print(f"  {index_dir / 'video_embeddings.npy'}")
    print(f"Videos indexed: {video_embeds.shape[0]} | Embed dim: {video_embeds.shape[1]}")


if __name__ == "__main__":
    main()
