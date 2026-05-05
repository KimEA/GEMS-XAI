"""
Module 4: XAI Analyzer
=======================
EdgeSHAPer를 GEMS 및 GCN 앙상블에 맞게 적응한 XAI 분석 모듈.

핵심 기능:
    1. EdgeSHAPer4GEMS: GEMS/GCN 모델에서 각 엣지의 Shapley 값 계산
    2. classify_edge:   엣지 타입 분류 (Ligand / Interaction / Protein / Self-loop)
    3. compute_topk_stats: k=5,10,15,20,25 별 Top-k 엣지 타입 비율 계산
    4. run_xai_for_group: 친화도 그룹 전체에 대한 XAI 일괄 실행

엣지 분류 기준 (GEMS B6AEPL 그래프 기준):
    - n_lig = n_nodes[1]: 리간드 원자 수
    - Ligand   edge: src < n_lig  AND dst < n_lig  AND src ≠ dst  (공유결합)
    - Interact edge: (src < n_lig AND dst ≥ n_lig) OR (src ≥ n_lig AND dst < n_lig)  (비공유 상호작용)
    - Protein  edge: src ≥ n_lig AND dst ≥ n_lig  AND src ≠ dst  (단백질 내부, GEMS에는 없음)
    - Self-loop:     src == dst  (제외)
"""

import os
import sys
import csv
import json
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

GEMS_DIR = os.path.join(os.path.dirname(__file__), "..", "GEMS")
sys.path.insert(0, GEMS_DIR)

# ─── 상수 ────────────────────────────────────────────────────────────────────

EDGE_TYPES = ["ligand", "interaction", "protein", "self_loop"]

# Top-k 분석에 사용할 k 값 목록
DEFAULT_K_VALUES = [5, 10, 15, 20, 25]


# ─── 단일 그래프 예측 헬퍼 ──────────────────────────────────────────────────────

def _make_single_graphbatch(x, edge_index, edge_attr, lig_emb, device):
    """
    단일 그래프를 PyG Batch 객체로 변환 (EdgeSHAPer 내부 사용).

    Parameters:
        x:          [N, node_dim]
        edge_index: [2, E]
        edge_attr:  [E, edge_dim]
        lig_emb:    [1, 384] 또는 None
        device:     연산 디바이스
    """
    data = Data(
        x=x.float(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.float(),
    )
    batch_obj = Batch.from_data_list([data])

    # lig_emb 처리: GEMS18d forward가 [num_graphs, 384] 형태를 기대
    if lig_emb is not None:
        emb = lig_emb.float()
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)   # [384] → [1, 384]
        elif emb.dim() == 3:
            emb = emb.squeeze(0)     # [1, 1, 384] → [1, 384]
        batch_obj.lig_emb = emb.to(device)

    return batch_obj.to(device)


@torch.no_grad()
def _ensemble_predict(models: list, x, edge_index, edge_attr, lig_emb, device) -> float:
    """
    앙상블(또는 단일) 모델로 스케일된 예측값 반환.
    엣지가 없는 경우 0.0 반환.
    """
    if edge_index.shape[1] == 0:
        return 0.0

    gb = _make_single_graphbatch(x, edge_index, edge_attr, lig_emb, device)
    preds = []
    for m in models:
        try:
            out = m(gb).view(-1)
            preds.append(out.item())
        except Exception:
            pass
    if not preds:
        return 0.0
    return float(np.mean(preds))


# ─── 엣지 분류 ────────────────────────────────────────────────────────────────

def classify_edge(src: int, dst: int, n_lig: int) -> str:
    """
    단일 엣지의 타입을 분류.

    Args:
        src:   소스 노드 인덱스
        dst:   목적지 노드 인덱스
        n_lig: 리간드 원자 수 (n_nodes[1])

    Returns:
        "ligand"      - 리간드 내부 공유결합
        "interaction" - 단백질-리간드 비공유 상호작용
        "protein"     - 단백질 내부 엣지 (GEMS에는 없음)
        "self_loop"   - 자기 자신으로의 엣지
    """
    if src == dst:
        return "self_loop"
    src_is_lig = src < n_lig
    dst_is_lig = dst < n_lig
    if src_is_lig and dst_is_lig:
        return "ligand"
    elif src_is_lig != dst_is_lig:  # 하나만 리간드 원자
        return "interaction"
    else:
        return "protein"


def classify_all_edges(edge_index: torch.Tensor, n_lig: int) -> List[str]:
    """
    그래프의 모든 엣지 분류 (Self-loop 포함).

    Args:
        edge_index: [2, E]
        n_lig:      리간드 원자 수

    Returns:
        edge_types: 길이 E의 문자열 리스트
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    return [classify_edge(int(s), int(d), n_lig) for s, d in zip(src, dst)]


def edge_type_summary(edge_types: List[str]) -> dict:
    """엣지 타입별 개수 요약."""
    from collections import Counter
    cnt = Counter(edge_types)
    total = len(edge_types)
    return {
        t: {"count": cnt.get(t, 0), "pct": cnt.get(t, 0) / total * 100 if total > 0 else 0.0}
        for t in EDGE_TYPES
    }


# ─── EdgeSHAPer (GEMS/GCN 적응 버전) ─────────────────────────────────────────

class EdgeSHAPer4GEMS:
    """
    GEMS 및 SimpleGCN 앙상블 모델용 EdgeSHAPer.

    원본 EdgeSHAPer (AndMastro/protein-ligand-GNN)와 차이점:
        - model(x, edge_index, batch, edge_weight) 대신
          GEMS graphbatch (x, edge_index, edge_attr, lig_emb) 포맷 사용
        - 엣지 마스킹 시 edge_attr도 동일하게 필터링
        - GEMS 앙상블(5개 모델)의 평균 예측 사용

    Shapley 값 계산 원리:
        φ_j = (1/M) Σ [v(S∪{j}) - v(S)]
        - S: 랜덤 엣지 부분집합
        - v(S): 엣지 집합 S만 사용했을 때의 모델 예측값
        - M: Monte Carlo 샘플링 횟수
    """

    def __init__(self, models: list, graph: Data, device):
        """
        Args:
            models: GEMS 앙상블 또는 [SimpleGCN] 리스트
            graph:  설명할 단일 그래프 (PyG Data 객체)
            device: 연산 디바이스
        """
        self.models     = models
        self.device     = device

        # 그래프 컴포넌트를 CPU에서 유지 (마스킹 연산은 CPU에서 빠름)
        self.x          = graph.x.cpu()
        self.edge_index = graph.edge_index.cpu()
        self.edge_attr  = graph.edge_attr.cpu()
        self.lig_emb    = graph.lig_emb.cpu() if (
            hasattr(graph, "lig_emb") and graph.lig_emb is not None) else None

        self.n_nodes    = self.x.shape[0]
        self.n_edges    = self.edge_index.shape[1]

        # 그래프 밀도 기반 배경 그래프 생성 확률
        max_possible   = self.n_nodes * (self.n_nodes - 1)
        self.P         = self.n_edges / max_possible if max_possible > 0 else 0.5

    def explain(self, M: int = 50, seed: int = 42) -> List[float]:
        """
        전체 엣지에 대한 Shapley 값 계산.

        Args:
            M:    Monte Carlo 샘플링 횟수 (많을수록 정확, 느림)
            seed: 재현성을 위한 시드

        Returns:
            phi_edges: 길이 n_edges의 float 리스트
                       값이 클수록 해당 엣지가 예측에 중요함
        """
        rng       = default_rng(seed)
        phi_edges = []

        for j in tqdm(range(self.n_edges), desc="EdgeSHAPer", leave=False):
            marginal = 0.0
            for _ in range(M):
                # ─ 배경 그래프 z: 각 엣지 독립적으로 P 확률로 포함
                E_z = rng.binomial(1, self.P, self.n_edges).astype(bool)
                # ─ 랜덤 순열 π 생성
                pi = torch.randperm(self.n_edges).numpy()
                j_pos = int(np.where(pi == j)[0][0])

                # ─ S ∪ {j}: π에서 j 포함 이전까지는 그래프 G, 이후는 배경 z
                mask_plus = np.zeros(self.n_edges, dtype=bool)
                mask_minus = np.zeros(self.n_edges, dtype=bool)
                for k in range(self.n_edges):
                    edge_idx = pi[k]
                    if k <= j_pos:
                        mask_plus[edge_idx] = True
                    else:
                        mask_plus[edge_idx] = E_z[edge_idx]
                    # ─ S: j를 제외한 버전
                    if k < j_pos:
                        mask_minus[edge_idx] = True
                    else:
                        mask_minus[edge_idx] = E_z[edge_idx]

                # 마스크된 엣지 인덱스 및 특징 추출
                idx_plus  = torch.from_numpy(np.where(mask_plus)[0]).long()
                idx_minus = torch.from_numpy(np.where(mask_minus)[0]).long()

                ei_plus   = self.edge_index[:, idx_plus]
                ea_plus   = self.edge_attr[idx_plus]
                ei_minus  = self.edge_index[:, idx_minus]
                ea_minus  = self.edge_attr[idx_minus]

                # 모델 예측 (스케일된 값, 역정규화 불필요 - 차이만 계산)
                v_plus  = _ensemble_predict(
                    self.models, self.x, ei_plus,  ea_plus,  self.lig_emb, self.device)
                v_minus = _ensemble_predict(
                    self.models, self.x, ei_minus, ea_minus, self.lig_emb, self.device)

                marginal += (v_plus - v_minus)

            phi_edges.append(marginal / M)

        return phi_edges


# ─── Top-k 통계 ────────────────────────────────────────────────────────────────

def compute_topk_stats(
    phi_edges:  List[float],
    edge_types: List[str],
    k_values:   List[int] = None,
) -> Dict[int, Dict[str, float]]:
    """
    k값별로 Top-k 중요 엣지의 타입 비율 계산.
    Self-loop 엣지는 분석에서 제외됨.

    Args:
        phi_edges:  각 엣지의 Shapley 값 (길이 n_edges)
        edge_types: 각 엣지의 타입 (길이 n_edges)
        k_values:   분석할 k 값 리스트 (기본: [5, 10, 15, 20, 25])

    Returns:
        {k: {"ligand": 비율, "interaction": 비율, "protein": 비율, "n_valid": 유효 엣지수}}

    예시:
        k=10일 때 {"ligand": 0.3, "interaction": 0.7, "protein": 0.0, "n_valid": 10}
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    # Self-loop 제외 후 유효 엣지만 사용
    valid_idx = [i for i, t in enumerate(edge_types) if t != "self_loop"]
    valid_phi = [abs(phi_edges[i]) for i in valid_idx]
    valid_types = [edge_types[i] for i in valid_idx]

    results = {}
    for k in k_values:
        actual_k = min(k, len(valid_phi))
        if actual_k == 0:
            results[k] = {t: 0.0 for t in ["ligand", "interaction", "protein"]}
            results[k]["n_valid"] = 0
            continue

        # 절댓값 기준 내림차순 정렬 후 Top-k 선택
        sorted_idx = np.argsort(valid_phi)[::-1][:actual_k]
        topk_types = [valid_types[i] for i in sorted_idx]

        from collections import Counter
        cnt = Counter(topk_types)
        results[k] = {
            t: cnt.get(t, 0) / actual_k for t in ["ligand", "interaction", "protein"]
        }
        results[k]["n_valid"] = actual_k

    return results


# ─── 그룹별 XAI 분석 ─────────────────────────────────────────────────────────

def run_xai_for_group(
    models:       list,
    graphs:       List[Data],
    group_name:   str,
    model_name:   str,
    M:            int = 50,
    k_values:     List[int] = None,
    output_dir:   str = None,
    seed:         int = 42,
    device=None,
    max_samples:  Optional[int] = None,
) -> dict:
    """
    특정 친화도 그룹의 모든 샘플에 EdgeSHAPer를 실행하고 통계 집계.

    Args:
        models:      모델 리스트 (GEMS 앙상블 또는 [SimpleGCN])
        graphs:      해당 그룹의 그래프 리스트
        group_name:  그룹 이름 ("low", "medium", "high")
        model_name:  모델 이름 (결과 파일명에 사용)
        M:           Monte Carlo 샘플링 횟수
        k_values:    Top-k 분석 k 목록
        output_dir:  결과 저장 디렉터리
        seed:        랜덤 시드
        device:      연산 디바이스
        max_samples: 최대 샘플 수 (None이면 전체 사용)

    Returns:
        group_stats: {
            "per_sample": [각 샘플의 topk_stats 딕셔너리],
            "mean_topk":  {k: {type: mean_ratio}},   # k별 평균 비율
            "n_samples":  분석한 샘플 수,
            "group":      group_name,
        }
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES
    if device is None:
        device = torch.device("cpu")
    if max_samples:
        graphs = graphs[:max_samples]

    print(f"\n[XAI] ── [{model_name}] {group_name} 그룹 XAI 분석 ──")
    print(f"      샘플 수={len(graphs)}, M={M}, k_values={k_values}")

    per_sample_stats = []

    for graph in tqdm(graphs, desc=f"XAI [{group_name}]"):
        # 엣지 타입 분류 (Self-loop 포함)
        n_lig      = int(graph.n_nodes[1].item())
        edge_types = classify_all_edges(graph.edge_index, n_lig)

        n_edges = graph.edge_index.shape[1]
        n_lig_e = sum(1 for t in edge_types if t == "ligand")
        n_int_e = sum(1 for t in edge_types if t == "interaction")
        n_prot_e = sum(1 for t in edge_types if t == "protein")

        # EdgeSHAPer 실행
        explainer = EdgeSHAPer4GEMS(models, graph, device)
        phi_edges = explainer.explain(M=M, seed=seed)

        # Top-k 통계
        topk_stats = compute_topk_stats(phi_edges, edge_types, k_values)

        sample_result = {
            "id":          graph.id,
            "pki":         float(graph.y.item() * 16),
            "n_edges":     n_edges,
            "n_lig":       n_lig_e,
            "n_interaction": n_int_e,
            "n_protein":   n_prot_e,
            "topk_stats":  topk_stats,
        }
        per_sample_stats.append(sample_result)

        # 샘플별 결과 저장
        if output_dir:
            sample_dir = os.path.join(output_dir, model_name, group_name, graph.id)
            os.makedirs(sample_dir, exist_ok=True)

            # Shapley 값 CSV
            edge_csv = os.path.join(sample_dir, f"{graph.id}_shapley.csv")
            src = graph.edge_index[0].cpu().numpy()
            dst = graph.edge_index[1].cpu().numpy()
            with open(edge_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["edge_idx", "src", "dst", "type", "shapley", "abs_shapley"])
                for i, (s, d, t, phi) in enumerate(zip(src, dst, edge_types, phi_edges)):
                    w.writerow([i, s, d, t, f"{phi:.6f}", f"{abs(phi):.6f}"])

    # ─ 그룹 집계: k별 평균 비율 계산
    mean_topk = {}
    for k in k_values:
        lig_ratios  = [s["topk_stats"][k]["ligand"]      for s in per_sample_stats]
        int_ratios  = [s["topk_stats"][k]["interaction"]  for s in per_sample_stats]
        prot_ratios = [s["topk_stats"][k]["protein"]      for s in per_sample_stats]
        mean_topk[k] = {
            "ligand":      float(np.mean(lig_ratios)),
            "interaction": float(np.mean(int_ratios)),
            "protein":     float(np.mean(prot_ratios)),
            "std_ligand":  float(np.std(lig_ratios)),
            "std_interaction": float(np.std(int_ratios)),
        }

    group_stats = {
        "group":       group_name,
        "model":       model_name,
        "n_samples":   len(per_sample_stats),
        "per_sample":  per_sample_stats,
        "mean_topk":   mean_topk,
    }

    # 그룹 요약 출력
    print(f"\n  ── [{group_name}] Top-k 엣지 타입 평균 비율 ──")
    print(f"  {'k':>4} | {'Ligand':>8} | {'Interaction':>12} | {'Protein':>8}")
    print(f"  {'-'*44}")
    for k in k_values:
        d = mean_topk[k]
        print(f"  {k:>4} | {d['ligand']*100:>7.1f}% | "
              f"{d['interaction']*100:>11.1f}% | {d['protein']*100:>7.1f}%")

    # 그룹 요약 JSON 저장
    if output_dir:
        summary_path = os.path.join(output_dir, model_name, f"{group_name}_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        # per_sample은 크기가 크므로 topk_stats만 직렬화
        save_data = {
            "group": group_stats["group"],
            "model": group_stats["model"],
            "n_samples": group_stats["n_samples"],
            "mean_topk": group_stats["mean_topk"],
            "per_sample_ids": [s["id"] for s in per_sample_stats],
            "per_sample_pki": [s["pki"] for s in per_sample_stats],
        }
        with open(summary_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n[XAI] 그룹 요약 저장: {summary_path}")

    return group_stats


# ─── 전체 XAI 파이프라인 ─────────────────────────────────────────────────────

def run_full_xai_analysis(
    wrappers:     dict,
    val_dataset,
    group_indices: dict,
    M:            int = 50,
    k_values:     List[int] = None,
    output_dir:   str = None,
    max_per_group: Optional[int] = None,
    device=None,
) -> dict:
    """
    여러 모델 × 여러 친화도 그룹에 대해 XAI 분석 일괄 실행.

    Args:
        wrappers:       {모델명: ModelWrapper}
        val_dataset:    검증 Subset
        group_indices:  {그룹명: [local indices in val_dataset]}
        M:              Monte Carlo 샘플링 횟수
        k_values:       Top-k 분석 k 목록
        output_dir:     결과 저장 디렉터리
        max_per_group:  그룹당 최대 샘플 수 (None = 전체)
        device:         연산 디바이스

    Returns:
        all_results: {모델명: {그룹명: group_stats}}
    """
    from .data_loader import get_graphs_from_indices

    if k_values is None:
        k_values = DEFAULT_K_VALUES

    all_results = {}

    for model_name, wrapper in wrappers.items():
        all_results[model_name] = {}
        for group_name, local_idxs in group_indices.items():
            graphs = get_graphs_from_indices(val_dataset, local_idxs)
            stats  = run_xai_for_group(
                models      = wrapper.models,
                graphs      = graphs,
                group_name  = group_name,
                model_name  = model_name,
                M           = M,
                k_values    = k_values,
                output_dir  = output_dir,
                device      = device,
                max_samples = max_per_group,
            )
            all_results[model_name][group_name] = stats

    return all_results
