from torch.utils.data import DataLoader
from data.video_text_dataset import MSVDDataset


def main():
    ds = MSVDDataset(split="val", num_frames=4)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(dl))
    print("TYPE:", type(batch))

    if isinstance(batch, dict):
        print("KEYS:", list(batch.keys()))
        for k, v in batch.items():
            shape = getattr(v, "shape", None)
            print(f"{k}: {type(v)} shape={shape}")
    else:
        print("BATCH (raw):", batch)
        try:
            frames, captions = batch
            print("frames type:", type(frames), "shape:", getattr(frames, "shape", None))
            print("captions type:", type(captions))
        except Exception as e:
            print("Could not unpack batch:", e)


if __name__ == "__main__":
    main()
