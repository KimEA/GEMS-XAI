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

GEMS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "GEMS")
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
    preds = [m(gb).view(-1).item() for m in models]
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
                # ─ 랜덤 순열 π 생성 (numpy rng으로 통일 — torch 전역 RNG 오염 방지)
                pi = rng.permutation(self.n_edges)
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
        {k: {
            "ligand":      Top-k 내 비율,
            "interaction": Top-k 내 비율,
            "protein":     Top-k 내 비율,
            "n_valid":     유효 엣지 수,
            "baseline_ligand":      그래프 전체 내 비율 (self-loop 제외),
            "baseline_interaction": 그래프 전체 내 비율,
            "baseline_protein":     그래프 전체 내 비율,
            "lift_ligand":      Top-k비율 / baseline (1이면 랜덤과 동일),
            "lift_interaction": Top-k비율 / baseline,
            "lift_protein":     Top-k비율 / baseline,
        }}

    Lift 해석:
        lift > 1 : 모델이 해당 엣지 타입을 기대 이상으로 중시
        lift ≈ 1 : 엣지 구성 비율과 동일한 수준 (무관심)
        lift < 1 : 해당 엣지 타입을 기대 이하로 중시
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    from collections import Counter

    # Self-loop 제외 후 유효 엣지만 사용
    valid_idx   = [i for i, t in enumerate(edge_types) if t != "self_loop"]
    valid_phi   = [abs(phi_edges[i]) for i in valid_idx]
    valid_types = [edge_types[i] for i in valid_idx]

    # ── 베이스라인: 그래프 전체 유효 엣지 구성 비율
    n_valid_total = len(valid_types)
    base_cnt  = Counter(valid_types)
    baseline  = {
        t: base_cnt.get(t, 0) / n_valid_total if n_valid_total > 0 else 0.0
        for t in ["ligand", "interaction", "protein"]
    }

    results = {}
    for k in k_values:
        actual_k = min(k, n_valid_total)
        if actual_k == 0:
            empty = {t: 0.0 for t in ["ligand", "interaction", "protein"]}
            results[k] = {
                **empty,
                "n_valid": 0,
                **{f"baseline_{t}": baseline[t] for t in ["ligand", "interaction", "protein"]},
                **{f"lift_{t}": float("nan") for t in ["ligand", "interaction", "protein"]},
            }
            continue

        # 절댓값 기준 내림차순 정렬 후 Top-k 선택
        sorted_idx = np.argsort(valid_phi)[::-1][:actual_k]
        topk_types = [valid_types[i] for i in sorted_idx]

        cnt = Counter(topk_types)
        topk_ratio = {t: cnt.get(t, 0) / actual_k for t in ["ligand", "interaction", "protein"]}

        # lift = Top-k 비율 / baseline 비율 (baseline=0이면 nan)
        lift = {
            t: topk_ratio[t] / baseline[t] if baseline[t] > 0 else float("nan")
            for t in ["ligand", "interaction", "protein"]
        }

        results[k] = {
            **topk_ratio,
            "n_valid":  actual_k,
            **{f"baseline_{t}": baseline[t] for t in ["ligand", "interaction", "protein"]},
            **{f"lift_{t}": lift[t]          for t in ["ligand", "interaction", "protein"]},
        }

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
    if max_samples and max_samples < len(graphs):
        rng_sub = default_rng(seed + 1)   # EdgeSHAPer 내부 시드(seed)와 분리
        perm    = rng_sub.permutation(len(graphs))
        graphs  = [graphs[i] for i in perm[:max_samples]]

    print(f"\n[XAI] ── [{model_name}] {group_name} 그룹 XAI 분석 ──")
    print(f"      샘플 수={len(graphs)}, M={M}, k_values={k_values}")

    per_sample_stats = []

    for local_idx, graph in enumerate(tqdm(graphs, desc=f"XAI [{group_name}]")):
        # 엣지 타입 분류 (Self-loop 포함)
        n_lig      = int(graph.n_nodes[1].item())
        edge_types = classify_all_edges(graph.edge_index, n_lig)

        n_edges = graph.edge_index.shape[1]
        n_lig_e = sum(1 for t in edge_types if t == "ligand")
        n_int_e = sum(1 for t in edge_types if t == "interaction")
        n_prot_e = sum(1 for t in edge_types if t == "protein")

        # EdgeSHAPer 실행: 샘플별 독립 시드로 MC 노이즈 간 상관 제거
        explainer = EdgeSHAPer4GEMS(models, graph, device)
        phi_edges = explainer.explain(M=M, seed=seed + local_idx)

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

        # lift: nan 샘플 제외 후 평균
        def _mean_lift(key):
            vals = [s["topk_stats"][k][key] for s in per_sample_stats]
            finite = [v for v in vals if not np.isnan(v)]
            return float(np.mean(finite)) if finite else float("nan")

        mean_topk[k] = {
            "ligand":           float(np.mean(lig_ratios)),
            "interaction":      float(np.mean(int_ratios)),
            "protein":          float(np.mean(prot_ratios)),
            "std_ligand":       float(np.std(lig_ratios)),
            "std_interaction":  float(np.std(int_ratios)),
            "baseline_interaction": float(np.mean(
                [s["topk_stats"][k]["baseline_interaction"] for s in per_sample_stats])),
            "lift_ligand":      _mean_lift("lift_ligand"),
            "lift_interaction": _mean_lift("lift_interaction"),
            "lift_protein":     _mean_lift("lift_protein"),
        }

    group_stats = {
        "group":       group_name,
        "model":       model_name,
        "n_samples":   len(per_sample_stats),
        "per_sample":  per_sample_stats,
        "mean_topk":   mean_topk,
    }

    # 그룹 요약 출력
    print(f"\n  ── [{group_name}] Top-k 엣지 타입 평균 비율 및 Lift ──")
    print(f"  {'k':>4} | {'Interact%':>10} | {'Baseline%':>10} | {'Lift':>6}")
    print(f"  {'-'*40}")
    for k in k_values:
        d = mean_topk[k]
        lift_str = f"{d['lift_interaction']:.2f}" if not np.isnan(d['lift_interaction']) else " nan"
        print(f"  {k:>4} | {d['interaction']*100:>9.1f}% | "
              f"{d['baseline_interaction']*100:>9.1f}% | {lift_str:>6}")

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

    # 모델 키만 추출해서 통계 검정에 넘김 → all_results 원본 오염 방지
    model_keys = [k for k in all_results if not k.startswith("__")]
    if len(model_keys) > 1:
        stat_view = {k: all_results[k] for k in model_keys}
        stat_results = compare_groups_statistically(
            all_results = stat_view,
            k_values    = k_values,
            output_dir  = output_dir,
        )
        # __stat_tests__ 는 호출자가 직접 팝(pop)하지 않아도 되도록 삽입하지 않음
        # 필요하면 stat_results를 별도 저장
    return all_results


# ─── 통계 검정 ────────────────────────────────────────────────────────────────

def compare_groups_statistically(
    all_results: dict,
    k_values:   List[int] = None,
    output_dir: str = None,
) -> dict:
    """
    6개 모델 × 3개 친화도 그룹에 대해 interaction 비율의 통계 검정 수행.

    검정 구조:
        1. within-model: 각 모델에서 low/medium/high 그룹 간 Kruskal-Wallis
                         → 유의하면 쌍별 Mann-Whitney U + Bonferroni 보정
        2. between-model: 같은 그룹에서 6개 모델 간 Kruskal-Wallis
                          → 유의하면 쌍별 Mann-Whitney U + Bonferroni 보정

    효과 크기: rank-biserial correlation  r = 1 - 2U / (n1 * n2)
        |r| < 0.1: negligible, 0.1~0.3: small, 0.3~0.5: medium, >0.5: large

    Args:
        all_results: run_full_xai_analysis() 반환값 ({모델명: {그룹명: group_stats}})
                     "__stat_tests__" 키는 자동으로 제외됨.
        k_values:    분석할 k 값 목록 (None이면 DEFAULT_K_VALUES)
        output_dir:  결과 저장 디렉터리

    Returns:
        {
          "within_model":  {model_name: {k: {kw: {...}, pairwise: {...}}}},
          "between_model": {group_name: {k: {kw: {...}, pairwise: {...}}}},
        }
    """
    from scipy import stats as scipy_stats
    from itertools import combinations

    if k_values is None:
        k_values = DEFAULT_K_VALUES

    GROUP_ORDER  = ["low", "medium", "high"]
    model_names  = [m for m in all_results if m != "__stat_tests__"]

    def _bh_correction(p_values: list, alpha: float = 0.05) -> list:
        """
        Benjamini-Hochberg (BH) FDR 보정.
        검정 족(family) 내 p-value 리스트 → 보정 후 유의 여부 bool 리스트.
        동일 순서 보장.
        """
        n = len(p_values)
        if n == 0:
            return []
        arr = np.array(p_values, dtype=float)
        sorted_idx = np.argsort(arr)
        sorted_p   = arr[sorted_idx]
        bh_crit    = np.arange(1, n + 1) / n * alpha
        sig_mask   = sorted_p <= bh_crit
        reject     = np.zeros(n, dtype=bool)
        if np.any(sig_mask):
            cutoff = int(np.where(sig_mask)[0][-1])
            reject[sorted_idx[:cutoff + 1]] = True
        return reject.tolist()

    def _get_values(all_results, model_name, group_name, k, field):
        """모델+그룹+k에 해당하는 샘플별 field 값 리스트 (nan 제외)."""
        group_stats = all_results.get(model_name, {}).get(group_name)
        if group_stats is None:
            return []
        vals = [
            s["topk_stats"][k][field]
            for s in group_stats["per_sample"]
            if k in s["topk_stats"]
        ]
        return [v for v in vals if not (isinstance(v, float) and np.isnan(v))]

    def _rank_biserial(u_stat, n1, n2):
        """Mann-Whitney U → rank-biserial correlation."""
        denom = n1 * n2
        return float(1 - 2 * u_stat / denom) if denom > 0 else 0.0

    def _effect_label(r):
        a = abs(r)
        if a < 0.1:   return "negligible"
        if a < 0.3:   return "small"
        if a < 0.5:   return "medium"
        return "large"

    def _jonckheere_trend(samples_dict: dict, group_order: list, alpha: float = 0.05):
        """
        Jonckheere-Terpstra 검정: 순서 있는 k그룹에서 단조 증가 트렌드 검정.
        H₀: 그룹 간 분포 동일
        H₁: group_order 순서대로 단조 증가 (alternative="greater")

        within-model 검정 전용 — between-model에는 순서가 없으므로 적용 불가.
        scipy >= 1.7.0 필요.
        """
        ordered = [samples_dict[g] for g in group_order
                   if g in samples_dict and len(samples_dict[g]) >= 2]
        if len(ordered) < 2:
            return {"skipped": "그룹 수 부족 (최소 2그룹 필요)"}
        try:
            res = scipy_stats.jonckheere_terpstra(ordered, alternative="greater")
            return {
                "J":          round(float(res.statistic), 4),
                "p":          round(float(res.pvalue),    6),
                "sig":        bool(res.pvalue < alpha),
                "n_groups":   len(ordered),
                "n_per_group": {g: len(samples_dict[g]) for g in group_order
                                if g in samples_dict},
                "conclusion": "단조 증가 트렌드 유의" if res.pvalue < alpha else "트렌드 없음",
            }
        except AttributeError:
            return {"skipped": "scipy < 1.7.0 — jonckheere_terpstra 미지원"}

    def _kw_and_pairwise(samples_dict: dict, alpha: float = 0.05):
        """
        samples_dict: {label: [ratio 리스트]}
        Kruskal-Wallis (방향성 없는 전체 차이) → 유의하면 쌍별 Mann-Whitney + Bonferroni.
        between-model 비교에 사용.
        """
        labels  = [l for l, v in samples_dict.items() if len(v) >= 2]
        groups  = [samples_dict[l] for l in labels]

        result = {}

        if len(groups) < 2:
            result["kruskal_wallis"] = {"skipped": "그룹 수 부족"}
            return result

        kw_stat, kw_p = scipy_stats.kruskal(*groups)
        result["kruskal_wallis"] = {
            "H":       round(float(kw_stat), 4),
            "p":       round(float(kw_p),    6),
            "sig":     kw_p < alpha,
            "n_groups": len(groups),
            "n_per_group": {l: len(samples_dict[l]) for l in labels},
        }

        pairs = list(combinations(labels, 2))
        n_pairs = len(pairs)

        pairwise = {}
        for a_lbl, b_lbl in pairs:
            a_vals = samples_dict[a_lbl]
            b_vals = samples_dict[b_lbl]
            if len(a_vals) < 2 or len(b_vals) < 2:
                continue
            u_stat, mw_p = scipy_stats.mannwhitneyu(
                a_vals, b_vals, alternative="two-sided"
            )
            r = _rank_biserial(u_stat, len(a_vals), len(b_vals))
            pairwise[f"{a_lbl}_vs_{b_lbl}"] = {
                "U":          round(float(u_stat), 2),
                "p":          round(float(mw_p),   6),
                "p_bonf":     round(float(mw_p * n_pairs), 6),
                "sig_bonf":   (mw_p * n_pairs) < alpha,
                "r":          round(r, 4),
                "effect":     _effect_label(r),
                "mean_a":     round(float(np.mean(a_vals)), 4),
                "mean_b":     round(float(np.mean(b_vals)), 4),
            }
        result["pairwise"] = pairwise
        return result

    def _wilcoxon_vs_one(vals: list, alpha: float = 0.05) -> dict:
        """
        H₀: 샘플별 lift 중앙값 = 1.0  (모델이 랜덤과 다를 바 없음)
        H₁: 중앙값 > 1.0              (interaction을 기대 이상으로 선택)

        Wilcoxon 부호 순위 검정 (one-sample, one-sided greater).
        lift - 1.0 의 차이가 모두 0이면 검정 불가 → skipped 반환.
        """
        if len(vals) < 5:
            return {"skipped": f"샘플 수 부족 ({len(vals)}개)"}
        diffs = [v - 1.0 for v in vals]
        if all(d == 0 for d in diffs):
            return {"skipped": "모든 lift=1.0 (분산 없음)"}
        try:
            stat, p = scipy_stats.wilcoxon(diffs, alternative="greater")
        except ValueError as e:
            return {"skipped": str(e)}
        median_lift = float(np.median(vals))
        return {
            "W":           round(float(stat), 2),
            "p":           round(float(p),    6),
            "sig":         p < alpha,
            "n":           len(vals),
            "median_lift": round(median_lift, 4),
            "conclusion":  "lift > 1.0 유의" if p < alpha else "랜덤과 차이 없음",
        }

    # ── 0. Lift vs 1.0: 모델×그룹별 "랜덤 대비 유의성" 검정 ─────────────────
    lift_vs_one = {}
    for model_name in model_names:
        lift_vs_one[model_name] = {}
        for k in k_values:
            lift_vs_one[model_name][k] = {}
            for g in GROUP_ORDER:
                if g not in all_results.get(model_name, {}):
                    continue
                vals = _get_values(all_results, model_name, g, k, "lift_interaction")
                lift_vs_one[model_name][k][g] = _wilcoxon_vs_one(vals)

    # ── 1. Within-model: low → medium → high 단조 증가 트렌드 검정 ──────────────
    # 연구 가설: 친화도(↑) → interaction 비율(↑)  [단조 증가]
    # 주 검정: Jonckheere-Terpstra (H₁: 단조 증가, one-sided greater)
    # 보조 검정: Kruskal-Wallis (omnibus, 방향성 없음) + 사후 Mann-Whitney
    within_model = {}
    for model_name in model_names:
        within_model[model_name] = {}
        for k in k_values:
            ratio_samples = {
                g: _get_values(all_results, model_name, g, k, "interaction")
                for g in GROUP_ORDER
                if g in all_results.get(model_name, {})
            }
            lift_samples = {
                g: _get_values(all_results, model_name, g, k, "lift_interaction")
                for g in GROUP_ORDER
                if g in all_results.get(model_name, {})
            }
            kw_ratio  = _kw_and_pairwise(ratio_samples)
            kw_lift   = _kw_and_pairwise(lift_samples)
            within_model[model_name][k] = {
                "ratio": {
                    "jonckheere":    _jonckheere_trend(ratio_samples, GROUP_ORDER),
                    "kruskal_wallis": kw_ratio.get("kruskal_wallis", {}),
                    "pairwise":       kw_ratio.get("pairwise", {}),
                },
                "lift": {
                    "jonckheere":    _jonckheere_trend(lift_samples, GROUP_ORDER),
                    "kruskal_wallis": kw_lift.get("kruskal_wallis", {}),
                    "pairwise":       kw_lift.get("pairwise", {}),
                },
            }

    # ── FDR 보정 (lift_vs_one): 각 (model, group)에 대해 k 값들 간 BH 보정 ──
    for model_name in model_names:
        for g in GROUP_ORDER:
            valid_k = [
                k for k in k_values
                if g in lift_vs_one[model_name].get(k, {})
                and "p" in lift_vs_one[model_name][k].get(g, {})
            ]
            p_vals = [lift_vs_one[model_name][k][g]["p"] for k in valid_k]
            if p_vals:
                reject = _bh_correction(p_vals)
                for k, rej in zip(valid_k, reject):
                    lift_vs_one[model_name][k][g]["sig_fdr"] = bool(rej)

    # ── FDR 보정 (within_model JT): 각 model의 ratio/lift에 대해 k 값들 간 BH 보정 ──
    for model_name in model_names:
        for metric in ["ratio", "lift"]:
            valid_k = [
                k for k in k_values
                if "p" in within_model[model_name][k][metric].get("jonckheere", {})
            ]
            p_vals = [within_model[model_name][k][metric]["jonckheere"]["p"] for k in valid_k]
            if p_vals:
                reject = _bh_correction(p_vals)
                for k, rej in zip(valid_k, reject):
                    within_model[model_name][k][metric]["jonckheere"]["sig_fdr"] = bool(rej)

    # ── 2. Between-model: 같은 그룹에서 모델 간 비교 ────────────────────────
    between_model = {}
    all_groups = set()
    for m in model_names:
        all_groups.update(all_results[m].keys())
    all_groups -= {"__stat_tests__"}

    for group_name in sorted(all_groups):
        between_model[group_name] = {}
        for k in k_values:
            ratio_samples = {
                m: _get_values(all_results, m, group_name, k, "interaction")
                for m in model_names
                if group_name in all_results.get(m, {})
            }
            lift_samples = {
                m: _get_values(all_results, m, group_name, k, "lift_interaction")
                for m in model_names
                if group_name in all_results.get(m, {})
            }
            between_model[group_name][k] = {
                "ratio": _kw_and_pairwise(ratio_samples),
                "lift":  _kw_and_pairwise(lift_samples),
            }

    stat_results = {
        "lift_vs_one":   lift_vs_one,
        "within_model":  within_model,
        "between_model": between_model,
    }

    # ── 출력 요약 ──────────────────────────────────────────────────────────────
    print("\n[StatTest] ═══ 통계 검정 요약 ═══")

    # 0. Lift vs 1.0
    print("\n  [0] Lift > 1.0 검정 (Wilcoxon, H₁: 중앙값 > 1.0, 랜덤 대비 유의성)")
    print(f"  {'모델':35s} | {'그룹':8s} | {'k':>4} | {'median lift':>11} | {'p':>8} | 결론")
    print(f"  {'-'*85}")
    for m in model_names:
        for g in GROUP_ORDER:
            for k in k_values:
                res = lift_vs_one.get(m, {}).get(k, {}).get(g, {})
                if not res or "skipped" in res:
                    continue
                sig = "★" if res.get("sig") else " "
                print(f"  {sig} {m:35s} | {g:8s} | {k:>4} | "
                      f"{res['median_lift']:>11.3f} | {res['p']:>8.4f} | {res['conclusion']}")

    # 1. Within-model (JT 트렌드 검정 결과 표시)
    k_rep = k_values[len(k_values) // 2]   # 대표 k (중간값)
    print(f"\n  [1] Within-model: 단조 증가 트렌드 (JT, H₁: low≤mid≤high, 대표 k={k_rep})")
    print(f"  {'모델':35s} | JT ratio p | JT lift p | 결론(ratio)")
    print(f"  {'-'*75}")
    for m in model_names:
        jt_r = within_model[m][k_rep]["ratio"].get("jonckheere", {})
        jt_l = within_model[m][k_rep]["lift"].get("jonckheere", {})
        sig  = "★" if jt_r.get("sig") or jt_l.get("sig") else " "
        p_r  = jt_r.get("p", "?")
        p_l  = jt_l.get("p", "?")
        conc = jt_r.get("conclusion", jt_r.get("skipped", "?"))
        print(f"  {sig} {m:35s}  {str(p_r):>10}  {str(p_l):>9}  {conc}")

    print(f"\n  [2] Between-model: 모델 간 차이 (대표 k={k_rep})")
    print(f"  {'그룹':8s} | ratio p | lift p")
    print(f"  {'-'*35}")
    for g in GROUP_ORDER:
        if g not in between_model:
            continue
        kw_r = between_model[g][k_rep]["ratio"].get("kruskal_wallis", {})
        kw_l = between_model[g][k_rep]["lift"].get("kruskal_wallis", {})
        sig  = "★" if kw_r.get("sig") or kw_l.get("sig") else " "
        print(f"  {sig} {g:8s}  {kw_r.get('p','?'):>7}  {kw_l.get('p','?'):>7}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "stat_tests.json")
        with open(out_path, "w") as f:
            json.dump(stat_results, f, indent=2, ensure_ascii=False)
        print(f"\n[StatTest] 검정 결과 저장: {out_path}")

    return stat_results
