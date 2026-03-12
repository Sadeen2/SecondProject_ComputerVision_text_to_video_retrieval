# utils/metrics.py
import numpy as np


def compute_ranks(similarity: np.ndarray, gt_video_index: np.ndarray) -> np.ndarray:
    """
    similarity: (N_text, N_video)  higher = more similar
    gt_video_index: (N_text,) each query's correct video column index

    Returns:
      ranks: (N_text,) 1-based rank of the correct video for each text query
    """
    N_text = similarity.shape[0]
    ranks = np.zeros(N_text, dtype=np.int64)

    for i in range(N_text):
        # sort videos for text i by descending similarity
        order = np.argsort(-similarity[i])
        gt = int(gt_video_index[i])
        # rank is position of gt in sorted order (1-based)
        ranks[i] = int(np.where(order == gt)[0][0]) + 1

    return ranks


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def median_rank(ranks: np.ndarray) -> float:
    return float(np.median(ranks))


def mean_rank(ranks: np.ndarray) -> float:
    return float(np.mean(ranks))


def mean_average_precision(similarity: np.ndarray, gt_video_index: np.ndarray) -> float:
    """
    For MSVD: usually one relevant video per text query.
    AP for query i = 1 / rank(gt) because only one relevant item.
    mAP = mean(AP_i).
    """
    ranks = compute_ranks(similarity, gt_video_index)
    ap = 1.0 / ranks.astype(np.float32)
    return float(np.mean(ap))
