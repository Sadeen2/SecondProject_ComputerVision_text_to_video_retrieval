# main.py
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer

from models.config import ModelConfig
from models.joint_embedding import DualEncoderModel
from utils.io_utils import (
    load_checkpoint,
    validate_index_files,
    load_json,
    load_numpy,
    l2_normalize_np,
)
from utils.metrics import (
    compute_ranks,
    recall_at_k,
    median_rank,
    mean_rank,
    mean_average_precision,
)

# -----------------------------
# Project paths (auto)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index"
DEFAULT_ANN_CSV = PROJECT_ROOT / "data" / "processed" / "annotations.csv"
DEFAULT_FRAMES_ROOT = PROJECT_ROOT / "data" / "frames"

DEFAULT_VALIDATE_SPLIT = "val"

DEFAULT_TOPK = 5
DEFAULT_VALIDATE_BATCH_SIZE = 64
DEFAULT_MAX_TEXT_LEN = 32

# Show 8 frames per retrieved video (2 rows x 4 cols)
FRAMES_PER_RANK = 8


# -----------------------------
# Helpers: input / exit
# -----------------------------
def _is_exit(s: str) -> bool:
    return s.strip().lower() == "exit"


def ask(prompt: str, default: Optional[str] = None) -> str:
    """
    Read input. If user types 'exit' -> raise SystemExit.
    If Enter and default provided -> return default.
    """
    s = input(prompt).strip()
    if _is_exit(s):
        raise SystemExit
    if s == "" and default is not None:
        return default
    return s


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_from_list_strict(
    title: str,
    items: List[str],
    default_index: int = 0,
    max_show: int = 20,
) -> str:
    """
    Strict menu:
    - Enter selects default
    - Only accept valid numeric index
    - invalid input -> re-prompt
    """
    if not items:
        raise RuntimeError(f"No items found for: {title}")

    while True:
        print(f"\n--- {title} ---")
        show_n = min(len(items), max_show)
        for i in range(show_n):
            mark = " (default)" if i == default_index else ""
            print(f"  [{i}] {items[i]}{mark}")
        if len(items) > show_n:
            print(f"  ... ({len(items) - show_n} more not shown)")

        s = ask(f"Choose index [default {default_index}]: ", default=str(default_index))

        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(items):
                return items[idx]

        print("[main] Invalid choice. Please enter a valid index, or type 'exit' to quit.")


def parse_yes_no_strict(s: str, default_yes: bool = True) -> bool:
    s = s.strip().lower()
    if s == "":
        return default_yes
    if s in ("y", "yes"):
        return True
    if s in ("n", "no"):
        return False
    raise ValueError("Invalid yes/no")


def ask_yes_no(prompt: str, default_yes: bool) -> bool:
    while True:
        s = ask(prompt, default="")
        try:
            return parse_yes_no_strict(s, default_yes=default_yes)
        except ValueError:
            print("[main] Please answer with 'y'/'yes' or 'n'/'no', or type 'exit' to quit.")


# -----------------------------
# Dataset helpers
# -----------------------------
def load_split_rows(ann_csv: Path, split: str) -> List[Dict[str, str]]:
    if not ann_csv.exists():
        raise FileNotFoundError(f"Missing annotations CSV: {ann_csv}")

    rows: List[Dict[str, str]] = []
    with ann_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("split", "").strip() == split:
                rows.append({
                    "video_id": r.get("video_id", "").strip(),
                    "caption": r.get("caption", "").strip(),
                })

    if not rows:
        raise RuntimeError(f"No rows for split='{split}' in {ann_csv}")
    return rows


def choose_query_caption_random_list(rows: List[Dict[str, str]], show_n: int = 100) -> str:
    """
    Show a RANDOM subset of captions each time (shuffled).
    User chooses index from shown subset or Enter for random.
    Strict: invalid -> re-prompt.
    """
    if not rows:
        raise RuntimeError("Empty rows for caption selection.")

    show_n = min(show_n, len(rows))
    idxs = np.random.choice(len(rows), size=show_n, replace=False)
    np.random.shuffle(idxs)
    subset = [rows[i] for i in idxs]

    while True:
        print("\n================ Query Selection ================")
        print("Choose a caption query from a RANDOM list (changes every time).")
        print("  - Enter caption index from this list")
        print("  - Press Enter for a RANDOM caption from this list")
        print("  - Type 'exit' to quit\n")

        for i, r in enumerate(subset):
            cap = r["caption"]
            cap_short = (cap[:90] + "...") if len(cap) > 90 else cap
            print(f"  [{i}] {cap_short}")

        s = ask("\nChoose caption index [default: random]: ", default="")

        if s == "":
            ridx = np.random.randint(0, len(subset))
            print(f"[main] Random caption index (from shown list) = {ridx}")
            return subset[ridx]["caption"]

        if s.isdigit():
            ci = int(s)
            if 0 <= ci < len(subset):
                return subset[ci]["caption"]

        print("[main] Invalid caption choice. Enter a valid index, or type 'exit' to quit.")


# -----------------------------
# Model/index loading
# -----------------------------
def detect_backbone_from_ckpt(ckpt: Dict) -> str:
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        return ckpt["cfg"].get("backbone_name", "resnet18")
    return ckpt.get("backbone", "resnet18")


def detect_text_model_from_ckpt(ckpt: Dict) -> str:
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        return ckpt["cfg"].get("text_model_name", "distilbert-base-uncased")
    return "distilbert-base-uncased"


def load_index(index_dir: Path) -> Tuple[Dict, List[str], np.ndarray]:
    paths = validate_index_files(index_dir)
    meta = load_json(paths["meta"])
    video_ids: List[str] = load_json(paths["video_ids"])
    video_embeds = load_numpy(paths["video_embeds"])  # (N_video, D)
    video_embeds = l2_normalize_np(video_embeds)
    return meta, video_ids, video_embeds


def load_model_and_tokenizer(checkpoint_path: Path, device: str, meta: Dict) -> Tuple[DualEncoderModel, AutoTokenizer, ModelConfig]:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    backbone = detect_backbone_from_ckpt(ckpt)
    text_model = detect_text_model_from_ckpt(ckpt)

    cfg = ModelConfig(
        backbone_name=backbone,
        text_model_name=text_model,
        num_frames=int(meta.get("num_frames", 4)),
        image_size=int(meta.get("image_size", 224)),
        device=device,
    )

    model = DualEncoderModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    return model, tokenizer, cfg


# -----------------------------
# Visualization: 8 frames per Rank (2x4) + tight layout
# -----------------------------
def list_frames_for_video(frames_root: Path, video_id: str) -> List[Path]:
    vid_dir = frames_root / video_id
    if not vid_dir.exists():
        return []
    return sorted(list(vid_dir.glob("*.jpg")))


def sample_k_frames(frames: List[Path], k: int) -> List[Path]:
    if not frames:
        return []
    if len(frames) >= k:
        idxs = np.linspace(0, len(frames) - 1, k).astype(int).tolist()
        return [frames[i] for i in idxs]
    out = frames[:]
    while len(out) < k:
        out.append(frames[-1])
    return out[:k]


def load_rgb_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def show_search_results(
    query_text: str,
    top_results: List[Tuple[str, float]],
    frames_root: Path,
    top_k: int,
    frames_per_rank: int = 8,
) -> None:
    """
    Requirements:
    - Caption title at the very top alone (not beside images).
    - For each Rank: show 8 frames in a compact 2x4 grid.
    - Frames should be larger, close together, and no big gaps.
    """
    k = min(top_k, len(top_results))
    if k == 0:
        print("[main] No results to display.")
        return

    if frames_per_rank != 8:
        frames_per_rank = 8  # force for this project request

    rows, cols = 2, 4  # 8 = 2x4

    # Bigger frames + tighter gaps:
    # - Increase dpi & size
    # - Use subgridspec with near-zero spacing
    fig = plt.figure(figsize=(3.2 * k, 6.0), dpi=130)
    fig.suptitle(query_text, fontsize=16, y=0.985)

    # Outer grid: one column per rank, very small spacing between ranks
    outer = fig.add_gridspec(nrows=1, ncols=k, wspace=0.06)

    for col_idx in range(k):
        video_id, score = top_results[col_idx]
        frames = list_frames_for_video(frames_root, video_id)
        sel = sample_k_frames(frames, frames_per_rank)

        # Inner grid inside rank cell: 2x4, almost no spacing
        inner = outer[col_idx].subgridspec(rows, cols, wspace=0.01, hspace=0.01)

        for j in range(frames_per_rank):
            ax = fig.add_subplot(inner[j // cols, j % cols])
            ax.axis("off")

            if j < len(sel):
                ax.imshow(load_rgb_image(sel[j]))
            else:
                ax.text(0.05, 0.5, "Frame not found", fontsize=8)

            # Put rank header ONCE (above the rank column)
            if j == 0:
                ax.set_title(
                    f"Rank {col_idx + 1}\n{video_id}\nscore={score:.3f}",
                    fontsize=10,
                    pad=6,
                )

    # Tight overall layout under the title
    plt.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.88, wspace=0.06)
    plt.show()


# -----------------------------
# Core: encoding / search
# -----------------------------
@torch.no_grad()
def encode_text(
    model: DualEncoderModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: str,
    max_text_len: int,
) -> np.ndarray:
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    t_emb = model.text_encoder(tokens)  # (B, D)
    t_emb = torch.nn.functional.normalize(t_emb, dim=1)
    return t_emb.detach().cpu().numpy()


def search_topk(
    query: str,
    model: DualEncoderModel,
    tokenizer: AutoTokenizer,
    video_ids: List[str],
    video_embeds: np.ndarray,
    device: str,
    top_k: int,
    max_text_len: int,
) -> List[Tuple[str, float]]:
    q_emb = encode_text(model, tokenizer, [query], device=device, max_text_len=max_text_len)[0]
    sims = video_embeds @ q_emb
    k = min(top_k, sims.shape[0])
    idx = np.argsort(-sims)[:k]
    return [(video_ids[i], float(sims[i])) for i in idx]


# -----------------------------
# Core: validate (multi-caption)
# -----------------------------
@torch.no_grad()
def run_validate(
    split_rows: List[Dict[str, str]],
    model: DualEncoderModel,
    tokenizer: AutoTokenizer,
    video_ids: List[str],
    video_embeds: np.ndarray,
    device: str,
    batch_size: int,
    max_text_len: int,
) -> Dict[str, float]:
    vid_to_idx = {vid: i for i, vid in enumerate(video_ids)}

    captions: List[str] = []
    gt_idx: List[int] = []
    for r in split_rows:
        vid = r["video_id"]
        cap = r["caption"]
        if vid in vid_to_idx and cap:
            captions.append(cap)
            gt_idx.append(vid_to_idx[vid])

    if not captions:
        raise RuntimeError(
            "No valid captions matched to index video_ids.\n"
            "Fix: build your index from the SAME dataset/split structure, and make sure video_ids.json matches annotations.csv video_id values."
        )

    gt_video_index = np.array(gt_idx, dtype=np.int64)

    text_embeds_list: List[np.ndarray] = []
    n = len(captions)
    for s in range(0, n, batch_size):
        batch_caps = captions[s:s + batch_size]
        emb = encode_text(model, tokenizer, batch_caps, device=device, max_text_len=max_text_len)
        text_embeds_list.append(emb)

    text_embeds = np.concatenate(text_embeds_list, axis=0)
    similarity = text_embeds @ video_embeds.T

    ranks = compute_ranks(similarity, gt_video_index)

    results = {
        "R@1": recall_at_k(ranks, 1),
        "R@5": recall_at_k(ranks, 5),
        "R@10": recall_at_k(ranks, 10),
        "MedR": median_rank(ranks),
        "MeanR": mean_rank(ranks),
        "mAP": mean_average_precision(similarity, gt_video_index),
        "N_text": float(len(captions)),
        "N_video": float(len(video_ids)),
    }
    return results


# -----------------------------
# MAIN
# -----------------------------
def print_banner():
    print("\n" + "=" * 60)
    print("Text-to-Video Retrieval System (Deep Learning) — MSVD")
    print("=" * 60)
    print("Method: Dual-Encoder (CNN video + Transformer text) + InfoNCE + Cached Video Index")
    print("\nRun modes:")
    print("  [1] Run (Search + optional Validate at end)")
    print("  [0] Exit")
    print("Type 'exit' at any prompt to quit.\n")


def main():
    try:
        print_banner()

        # strict mode choice
        while True:
            mode = ask("Choose mode [default 1]: ", default="1")
            if mode in ("0", "1"):
                break
            print("[main] Invalid mode. Enter 1 or 0, or type 'exit' to quit.")

        if mode == "0":
            print("[main] Bye.")
            return

        # ---- choose checkpoint (STRICT) ----
        ckpts = sorted([p.name for p in DEFAULT_CKPT_DIR.glob("*.pt")])
        ckpt_name = choose_from_list_strict("Choose checkpoint", ckpts, default_index=0)
        ckpt_path = DEFAULT_CKPT_DIR / ckpt_name

        # ---- choose index dir (STRICT) ----
        index_choices: List[str] = []
        if (DEFAULT_INDEX_DIR / "meta.json").exists():
            index_choices.append("index")
        for d in sorted([p for p in DEFAULT_INDEX_DIR.iterdir() if p.is_dir()]):
            if (d / "meta.json").exists():
                index_choices.append(f"index/{d.name}")

        if not index_choices:
            raise FileNotFoundError(f"No valid index found under: {DEFAULT_INDEX_DIR}")

        index_choice = choose_from_list_strict("Choose index directory", index_choices, default_index=0)
        index_dir = PROJECT_ROOT / index_choice

        # ---- device (STRICT-ish) ----
        device_default = auto_device()
        while True:
            dev = ask(f"Device [Enter for best='{device_default}'] (cpu/cuda): ", default=device_default)
            if dev in ("cpu", "cuda"):
                break
            print("[main] Invalid device. Please type 'cpu' or 'cuda', or 'exit' to quit.")

        # ---- load index + model ----
        meta, video_ids, video_embeds = load_index(index_dir)
        model, tokenizer, cfg = load_model_and_tokenizer(ckpt_path, device=dev, meta=meta)

        print("\n------------------------------------------------------------")
        print(f"Backbone: {cfg.backbone_name} | Text: {cfg.text_model_name}")
        print(f"num_frames: {cfg.num_frames} | image_size: {cfg.image_size} | device: {dev}")
        print(f"Index videos (unique): {len(video_ids)}")
        print("------------------------------------------------------------")

        # ----------------------------
        # SEARCH FIRST (always)
        # ----------------------------
        print("\n============================================================")
        print("Step: SEARCH (interactive)")
        print("============================================================")

        # top-k strict input
        while True:
            k_str = ask(f"\nTop-K [Enter={DEFAULT_TOPK}]: ", default=str(DEFAULT_TOPK))
            if k_str.isdigit() and int(k_str) >= 1:
                top_k = int(k_str)
                break
            print("[main] Invalid Top-K. Enter a positive integer, or type 'exit' to quit.")

        # captions from VAL for selection
        query_rows = load_split_rows(DEFAULT_ANN_CSV, "val")

        while True:
            print("\n------------------------------------------------------------")
            print("Search options:")
            print("  [1] Choose caption from RANDOM dataset list (index / random)")
            print("  [2] Type a custom query text")
            print("  [0] Finish Search (then optional Validate)")
            opt = ask("Choose option [default 1]: ", default="1")

            if opt not in ("0", "1", "2"):
                print("[main] Invalid option. Enter 0/1/2, or type 'exit' to quit.")
                continue

            if opt == "0":
                break

            if opt == "2":
                q = ask("Enter your query text [Enter=random caption]: ", default="")
                if q == "":
                    query = choose_query_caption_random_list(query_rows, show_n=100)
                else:
                    query = q
            else:
                query = choose_query_caption_random_list(query_rows, show_n=100)

            t0 = time.time()
            results = search_topk(
                query=query,
                model=model,
                tokenizer=tokenizer,
                video_ids=video_ids,
                video_embeds=video_embeds,
                device=dev,
                top_k=top_k,
                max_text_len=DEFAULT_MAX_TEXT_LEN,
            )
            t1 = time.time()

            print("\n🔎 Query:", query)
            print(f"Search time: {(t1 - t0):.4f} sec")
            print("\nTop-K results:")
            print("----------------------------------------")
            for rank, (vid, score) in enumerate(results, start=1):
                print(f"{rank:02d}) video_id={vid} | score={score:.4f}")

            # Show compact 2x4 frames per rank
            show_search_results(
                query_text=query,
                top_results=results,
                frames_root=DEFAULT_FRAMES_ROOT,
                top_k=top_k,
                frames_per_rank=FRAMES_PER_RANK,
            )

        # ----------------------------
        # ASK VALIDATE AT THE END ONLY
        # ----------------------------
        print("\n============================================================")
        print("Step: VALIDATE (optional, at the end)")
        print("============================================================")

        do_val = ask_yes_no("Run VALIDATE now? (y/n) [default n]: ", default_yes=False)

        if do_val:
            rows = load_split_rows(DEFAULT_ANN_CSV, DEFAULT_VALIDATE_SPLIT)

            t0 = time.time()
            results = run_validate(
                split_rows=rows,
                model=model,
                tokenizer=tokenizer,
                video_ids=video_ids,
                video_embeds=video_embeds,
                device=dev,
                batch_size=DEFAULT_VALIDATE_BATCH_SIZE,
                max_text_len=DEFAULT_MAX_TEXT_LEN,
            )
            t1 = time.time()

            print("\n📊 Text-to-Video Retrieval Results (MSVD multi-caption)")
            print("------------------------------------------------------")
            print(f"N_text queries: {int(results['N_text'])} | N_video unique: {int(results['N_video'])}")
            print(f"R@1   : {results['R@1']:.4f}")
            print(f"R@5   : {results['R@5']:.4f}")
            print(f"R@10  : {results['R@10']:.4f}")
            print(f"MedR  : {results['MedR']:.2f}")
            print(f"MeanR : {results['MeanR']:.2f}")
            print(f"mAP   : {results['mAP']:.4f}")
            print(f"Validate time: {(t1 - t0):.2f} sec")
        else:
            print("[main] Validate skipped.")

        print("\n[main] Done. Bye.")

    except SystemExit:
        print("\n[main] Exit requested.")
    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt -> quitting.")


if __name__ == "__main__":
    main()
