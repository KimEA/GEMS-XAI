"""
Module 1: Data Loader
======================
GEMS 포맷 PyTorch 데이터셋(.pt)을 로드하고,
80:20 학습/검증 분할 및 결합 친화도(pKi) 기준 층화(Stratification)를 수행.

B6AEPL 데이터셋 구조:
    - x: [N_nodes, 1148]  ← 60(원자) + 768(Ankh) + 320(ESM2)
    - edge_attr: [N_edges, 20]
    - edge_index: [2, N_edges]
    - lig_emb: [1, 384]  ← ChemBERTa-77M
    - n_nodes: [total, n_lig, n_prot]
    - y: float (0~1로 정규화된 pKi, 실제값 = y * 16)
    - id: str (PDB 코드)
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

# GEMS Dataset 클래스를 불러오기 위한 경로 추가
GEMS_DIR = os.path.join(os.path.dirname(__file__), "..", "GEMS")
sys.path.insert(0, GEMS_DIR)

# ─── 상수 ────────────────────────────────────────────────────────────────────
PK_MIN, PK_MAX = 0.0, 16.0  # GEMS 정규화 범위

# 결합 친화도 층화 기준 (pKi 단위)
# Low: pKi < 6.0  |  Medium: 6.5 ≤ pKi ≤ 7.5  |  High: pKi > 8.0
AFFINITY_THRESHOLDS = {
    "low":    {"max": 6.0},
    "medium": {"min": 6.5, "max": 7.5},
    "high":   {"min": 8.0},
}

# ─── 유틸리티 ─────────────────────────────────────────────────────────────────

def unscale_pk(scaled: float) -> float:
    """GEMS 정규화값(0~1) → 실제 pKi 값으로 역변환."""
    return float(scaled) * (PK_MAX - PK_MIN) + PK_MIN


def get_graph_pki(graph) -> float:
    """그래프 객체에서 실제 pKi 값 추출."""
    return unscale_pk(graph.y.item())


# ─── 데이터셋 로딩 ────────────────────────────────────────────────────────────

def load_gems_dataset(path: str):
    """
    GEMS 포맷 .pt 파일을 로드.

    Args:
        path: .pt 파일 경로 (예: 'B6AEPL_train_cleansplit.pt')

    Returns:
        PDBbind_Dataset 객체
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"데이터셋 파일이 없습니다: {path}")

    # weights_only=False: PDBbind_Dataset 클래스 역직렬화에 필요
    dataset = torch.load(path, weights_only=False)
    print(f"[DataLoader] 로드 완료: {os.path.basename(path)} ({len(dataset)} 샘플)")
    return dataset


def get_dataset_info(dataset) -> dict:
    """
    데이터셋 기본 정보를 딕셔너리로 반환.
    모델 초기화에 필요한 차원 정보도 포함.
    """
    sample = dataset[0]
    pki_values = [get_graph_pki(dataset[i]) for i in range(len(dataset))]

    info = {
        "n_samples":         len(dataset),
        "node_feat_dim":     sample.x.shape[1],
        "edge_feat_dim":     sample.edge_attr.shape[1],
        "lig_emb_dim":       sample.lig_emb.shape[-1] if (hasattr(sample, "lig_emb") and sample.lig_emb is not None) else 0,
        "protein_embeddings": getattr(dataset, "protein_embeddings", []),
        "ligand_embeddings":  getattr(dataset, "ligand_embeddings",  []),
        "pki_mean":   float(np.mean(pki_values)),
        "pki_std":    float(np.std(pki_values)),
        "pki_min":    float(np.min(pki_values)),
        "pki_max":    float(np.max(pki_values)),
    }
    print(f"[DataLoader] 데이터셋 정보:")
    for k, v in info.items():
        print(f"             {k}: {v}")
    return info


# ─── 데이터 분할 ──────────────────────────────────────────────────────────────

def split_train_val(dataset, train_ratio: float = 0.8, seed: int = 42):
    """
    데이터셋을 학습/검증으로 무작위 분할.

    Args:
        dataset:     GEMS PDBbind_Dataset
        train_ratio: 학습 비율 (기본 0.8 = 80%)
        seed:        재현성을 위한 랜덤 시드

    Returns:
        (train_subset, val_subset): torch.utils.data.Subset 쌍
    """
    n = len(dataset)
    indices = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx)

    print(f"[DataLoader] 분할 완료: 학습 {len(train_subset)} | 검증 {len(val_subset)}")
    return train_subset, val_subset


# ─── 층화 샘플링 ──────────────────────────────────────────────────────────────

def stratify_by_affinity(dataset_or_subset, thresholds: dict = None) -> dict:
    """
    결합 친화도(pKi) 기준으로 데이터를 3단계로 층화.

    Args:
        dataset_or_subset: GEMS 데이터셋 또는 Subset
        thresholds: 층화 기준 딕셔너리 (None이면 AFFINITY_THRESHOLDS 사용)

    Returns:
        {group_name: [indices_in_dataset_or_subset]} 딕셔너리
        - "low":    pKi < 6.0
        - "medium": 6.5 ≤ pKi ≤ 7.5
        - "high":   pKi > 8.0
    """
    if thresholds is None:
        thresholds = AFFINITY_THRESHOLDS

    groups = {name: [] for name in thresholds}

    # Subset인 경우 실제 인덱스 추출
    if isinstance(dataset_or_subset, Subset):
        base_dataset = dataset_or_subset.dataset
        indices      = dataset_or_subset.indices
    else:
        base_dataset = dataset_or_subset
        indices      = list(range(len(dataset_or_subset)))

    for local_idx, global_idx in enumerate(indices):
        graph = base_dataset[global_idx]
        pki   = get_graph_pki(graph)

        # 각 그룹 조건 확인
        for group_name, crit in thresholds.items():
            lo = crit.get("min", -float("inf"))
            hi = crit.get("max",  float("inf"))
            if lo <= pki <= hi if "min" in crit and "max" in crit else \
               pki < hi        if "max" in crit else \
               pki > lo:
                groups[group_name].append(local_idx)  # local index within subset
                break  # 각 샘플은 하나의 그룹에만 속함

    for name, idxs in groups.items():
        lo = thresholds[name].get("min", "-∞")
        hi = thresholds[name].get("max", "+∞")
        print(f"[DataLoader] [{name:6s}] pKi({lo}~{hi}): {len(idxs)} 샘플")

    return groups


def get_graphs_from_indices(dataset_or_subset, indices: list) -> list:
    """
    데이터셋(또는 Subset)에서 인덱스 리스트로 그래프 객체 추출.
    XAI 분석에 개별 그래프 객체가 필요할 때 사용.
    """
    if isinstance(dataset_or_subset, Subset):
        base_dataset = dataset_or_subset.dataset
        real_indices = [dataset_or_subset.indices[i] for i in indices]
    else:
        base_dataset = dataset_or_subset
        real_indices = indices

    return [base_dataset[i] for i in real_indices]


# ─── DataLoader 생성 ─────────────────────────────────────────────────────────

def create_dataloader(
    dataset_or_subset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    PyG DataLoader 생성. CPU 환경에서는 num_workers=0 권장 (MPS 호환성).

    Args:
        dataset_or_subset: GEMS 데이터셋 또는 Subset
        batch_size:  배치 크기
        shuffle:     셔플 여부
        num_workers: 병렬 로딩 프로세스 수 (CPU에서는 0 사용)

    Returns:
        DataLoader 객체
    """
    loader = DataLoader(
        dataset_or_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    print(f"[DataLoader] DataLoader 생성: batch={batch_size}, shuffle={shuffle}, "
          f"n_batches={len(loader)}")
    return loader


# ─── 빠른 확인용 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    dataset    = load_gems_dataset(args.dataset_path)
    info       = get_dataset_info(dataset)
    train, val = split_train_val(dataset, args.train_ratio)
    groups     = stratify_by_affinity(val)

    print("\n층화 결과 (검증 셋):")
    for g, idxs in groups.items():
        print(f"  {g}: {len(idxs)} 샘플")
