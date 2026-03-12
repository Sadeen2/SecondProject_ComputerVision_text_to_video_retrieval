# training/train.py
import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

from utils.seed import set_seed
from data.video_text_dataset import MSVDDataset, collate_fn
from models.config import ModelConfig
from models.joint_embedding import DualEncoderModel
from training.loss import InfoNCELoss


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, cfg, best_val_loss):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict() if scheduler is not None else None,
        "cfg": cfg.__dict__,
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, str(path))


def get_captions_from_batch(batch: dict):
    # Support both keys: caption / captions
    captions = batch.get("caption", None)
    if captions is None:
        captions = batch.get("captions", None)
    if captions is None:
        raise KeyError(f"Batch keys are {list(batch.keys())}, expected 'caption' or 'captions'")
    return captions


def run_one_epoch(model, tokenizer, loss_fn, loader, device, optimizer=None,
                  max_text_len=32, use_amp=False, grad_clip=1.0, scaler=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        video_frames = batch["video_frames"].to(device)  # [B, T, 3, H, W]
        captions = get_captions_from_batch(batch)         # list[str]

        tokens = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        if train_mode and use_amp:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                video_emb, text_emb = model(video_frames, tokens)
                loss = loss_fn(video_emb, text_emb)

            scaler.scale(loss).backward()

            # grad clipping (after unscale)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

        else:
            video_emb, text_emb = model(video_frames, tokens)
            loss = loss_fn(video_emb, text_emb)

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, tokenizer, loss_fn, loader, device, max_text_len=32):
    model.eval()
    return run_one_epoch(
        model=model,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        loader=loader,
        device=device,
        optimizer=None,
        max_text_len=max_text_len,
        use_amp=False,
        grad_clip=0.0,
        scaler=None,
    )


def main():
    ap = argparse.ArgumentParser(description="Step 3: Train dual-encoder with InfoNCE contrastive learning.")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")

    # data
    ap.add_argument("--annotations_csv", type=str, default="data/processed/annotations.csv")
    ap.add_argument("--frames_root", type=str, default="data/frames")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_text_len", type=int, default=32)

    # training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # model
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.07)

    ap.add_argument("--backbone", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--text_model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--freeze_text", action="store_true")

    # outputs
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")

    args = ap.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"\nUsing device: {device}")

    # ---- config ----
    cfg = ModelConfig()
    cfg.device = str(device)
    cfg.num_frames = args.num_frames
    cfg.image_size = args.image_size
    cfg.max_text_len = args.max_text_len
    cfg.embed_dim = args.embed_dim

    cfg.backbone_name = args.backbone
    cfg.freeze_backbone = args.freeze_backbone

    cfg.text_model_name = args.text_model
    cfg.freeze_text = args.freeze_text

    # ---- data ----
    train_ds = MSVDDataset(
        annotations_csv=args.annotations_csv,
        frames_root=args.frames_root,
        split="train",
        num_frames=args.num_frames,
        image_size=args.image_size,
    )
    val_ds = MSVDDataset(
        annotations_csv=args.annotations_csv,
        frames_root=args.frames_root,
        split="val",
        num_frames=args.num_frames,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    # ---- model ----
    model = DualEncoderModel(cfg).to(device)

    # ---- loss + optimizer ----
    loss_fn = InfoNCELoss(temperature=args.temperature)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- scheduler (improvement #1) ----
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- AMP scaler (improvement #2) ----
    use_amp = (args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- checkpoints ----
    ckpt_dir = Path(args.ckpt_dir)
    safe_mkdir(ckpt_dir)
    best_path = ckpt_dir / "model_best.pt"
    last_path = ckpt_dir / "model_last.pt"

    best_val_loss = float("inf")

    # ---- training loop ----
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = run_one_epoch(
            model=model,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            max_text_len=cfg.max_text_len,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            scaler=scaler,
        )

        val_loss = validate(
            model=model,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device,
            max_text_len=cfg.max_text_len,
        )

        scheduler.step()

        dt = time.time() - t0
        print(f"\nEpoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={dt:.1f}s")

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, cfg, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, cfg, best_val_loss)
            print(f"✅ Saved BEST checkpoint: {best_path} (val_loss={best_val_loss:.4f})")

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
