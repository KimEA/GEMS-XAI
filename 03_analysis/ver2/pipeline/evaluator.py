"""
Module 3: Evaluator
====================
GEMS 앙상블 및 SimpleGCN 모델 성능 평가 모듈.

평가 지표:
    - RMSE (pK 단위, 역정규화 후)
    - Pearson Correlation Coefficient (R)
    - R² (결정계수)

비교 실험 구성:
    ┌──────────────────┬──────────────────────┬──────────────────────┐
    │ 모델             │ 학습 데이터          │ 모델 파일            │
    ├──────────────────┼──────────────────────┼──────────────────────┤
    │ GEMS_PDBbind     │ PDBbind (전체)       │ 사용자 제공 경로     │
    │ GEMS_CleanSplit  │ CleanSplit (80%)     │ GEMS/model/ 기본 경로│
    │ GCN_PDBbind      │ B6AEPL_train_pdbbind │ 학습 후 저장된 .pt   │
    │ GCN_CleanSplit   │ B6AEPL_train_clean.. │ 학습 후 저장된 .pt   │
    └──────────────────┴──────────────────────┴──────────────────────┘

테스트셋: B6AEPL_casf2016.pt (CASF-2016, 282개 복합체)
"""

import os
import sys
import glob
import csv
import json
import torch
import numpy as np
import pandas as pd

GEMS_DIR = os.path.join(os.path.dirname(__file__), "..", "GEMS")
sys.path.insert(0, GEMS_DIR)

from torch_geometric.loader import DataLoader
from model.GEMS18 import GEMS18d, GEMS18e
from .trainer import SimpleGCN, load_gcn_checkpoint, RMSELoss
from .data_loader import unscale_pk

# ─── 상수 ────────────────────────────────────────────────────────────────────
PK_MIN, PK_MAX = 0.0, 16.0


# ─── 모델 래퍼 ────────────────────────────────────────────────────────────────

class ModelWrapper:
    """
    GEMS 앙상블 및 SimpleGCN 모델을 동일한 인터페이스로 래핑.
    evaluate_model() 함수에서 내부적으로 사용됨.

    Attributes:
        name:       모델 식별 이름 (예: 'GEMS_CleanSplit')
        models:     모델 리스트 (GEMS: 5개, GCN: 1개)
        is_ensemble: True이면 앙상블 평균, False이면 단일 모델
        device:     연산 디바이스
    """
    def __init__(self, name: str, models: list, device, is_ensemble: bool = True):
        self.name        = name
        self.models      = models
        self.is_ensemble = is_ensemble
        self.device      = device

    @torch.no_grad()
    def predict(self, graphbatch) -> torch.Tensor:
        """
        배치 입력으로부터 예측값(정규화 스케일) 반환.

        Returns:
            [B] 텐서 (배치 내 각 그래프의 예측값, 0~1 스케일)
        """
        graphbatch = graphbatch.to(self.device)
        if self.is_ensemble:
            preds = [m(graphbatch).view(-1) for m in self.models]
            return torch.mean(torch.stack(preds), dim=0)
        else:
            return self.models[0](graphbatch).view(-1)


# ─── GEMS 앙상블 로드 ─────────────────────────────────────────────────────────

def load_gems_ensemble(
    model_path: str,
    model_arch: str,
    dataset_id: str,
    node_feat_dim: int,
    edge_feat_dim: int,
    device,
) -> list:
    """
    5-fold GEMS 앙상블 모델 로드.

    Args:
        model_path:    모델 디렉터리 (stdict 파일들이 있는 곳)
        model_arch:    "GEMS18d" 또는 "GEMS18e"
        dataset_id:    "B6AEPL", "00AEPL" 등
        node_feat_dim: 노드 특징 차원
        edge_feat_dim: 엣지 특징 차원
        device:        torch.device

    Returns:
        models: [model_f0, model_f1, ..., model_f4] (최대 5개)
    """
    patterns = [
        os.path.join(model_path, f"{model_arch}_{dataset_id}_*_f{f}_best_stdict.pt")
        for f in range(5)
    ]
    stdict_paths = sorted([p for pat in patterns for p in glob.glob(pat)])

    if not stdict_paths:
        raise FileNotFoundError(
            f"모델 파일 없음: {model_arch}/{dataset_id} in {model_path}"
        )

    model_class = GEMS18d if model_arch == "GEMS18d" else GEMS18e
    models = []
    for path in stdict_paths:
        m = model_class(
            dropout_prob=0,
            in_channels=node_feat_dim,
            edge_dim=edge_feat_dim,
            conv_dropout_prob=0,
        ).float().to(device)
        state = torch.load(path, map_location=device, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    print(f"[Evaluator] GEMS 앙상블 로드: {len(models)}개 모델 ({model_arch}/{dataset_id})")
    return models


def auto_load_gems_ensemble(model_path: str, dataset, device) -> list:
    """
    데이터셋 메타정보를 읽어 자동으로 GEMS 앙상블 로드.
    주어진 데이터셋의 protein_embeddings, ligand_embeddings에 맞는
    model_arch와 dataset_id를 자동 결정.
    """
    node_feat_dim = dataset[0].x.shape[1]
    edge_feat_dim = dataset[0].edge_attr.shape[1]
    has_lig  = len(getattr(dataset, "ligand_embeddings",  [])) > 0
    has_prot = len(getattr(dataset, "protein_embeddings", [])) > 0
    prot_embs = getattr(dataset, "protein_embeddings", [])

    if not has_lig:
        model_arch, dataset_id = "GEMS18e", "00AEPL"
    elif not has_prot:
        model_arch, dataset_id = "GEMS18d", "00AEPL"
    elif "ankh_base" in prot_embs and "esm2_t6" in prot_embs:
        model_arch, dataset_id = "GEMS18d", "B6AEPL"
    elif "ankh_base" in prot_embs:
        model_arch, dataset_id = "GEMS18d", "B0AEPL"
    elif "esm2_t6" in prot_embs:
        model_arch, dataset_id = "GEMS18d", "06AEPL"
    else:
        raise ValueError(f"알 수 없는 임베딩 조합: {prot_embs}")

    return load_gems_ensemble(
        model_path, model_arch, dataset_id,
        node_feat_dim, edge_feat_dim, device
    )


# ─── 성능 지표 계산 ───────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    RMSE, Pearson R, R² 계산.

    Args:
        y_true: 실제 pK 값 배열 (역정규화 완료)
        y_pred: 예측 pK 값 배열 (역정규화 완료)

    Returns:
        {"RMSE": float, "R": float, "R2": float}
    """
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r    = float(np.corrcoef(y_true, y_pred)[0, 1])
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"RMSE": rmse, "R": r, "R2": r2}


# ─── 단일 모델 평가 ───────────────────────────────────────────────────────────

def evaluate_model(
    wrapper: ModelWrapper,
    test_dataset,
    batch_size: int = 64,
    save_csv: str = None,
) -> dict:
    """
    ModelWrapper를 사용해 테스트 데이터셋 전체 평가.

    Args:
        wrapper:      ModelWrapper 인스턴스
        test_dataset: 평가용 GEMS 데이터셋
        batch_size:   배치 크기
        save_csv:     예측값 저장 경로 (None이면 저장 안 함)

    Returns:
        metrics: {"RMSE": ..., "R": ..., "R2": ...}
    """
    loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = RMSELoss()

    y_true_list, y_pred_list, ids = [], [], []

    for batch in loader:
        y_pred_scaled = wrapper.predict(batch).cpu()
        y_true_scaled = batch.y.cpu()

        # pKi 역정규화 (0~1 → 실제 pKi)
        y_pred_list.append(y_pred_scaled * (PK_MAX - PK_MIN) + PK_MIN)
        y_true_list.append(y_true_scaled * (PK_MAX - PK_MIN) + PK_MIN)
        ids.extend(list(batch.id))

    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()

    metrics = compute_metrics(y_true, y_pred)
    print(f"[Evaluator] {wrapper.name:20s} | "
          f"RMSE={metrics['RMSE']:.4f} | R={metrics['R']:.4f} | R²={metrics['R2']:.4f}")

    # CSV 저장
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "y_true_pk", "y_pred_pk", "error"])
            for sid, yt, yp in sorted(zip(ids, y_true, y_pred)):
                writer.writerow([sid, f"{yt:.4f}", f"{yp:.4f}", f"{abs(yt-yp):.4f}"])
        print(f"[Evaluator] 예측값 저장: {save_csv}")

    return metrics


# ─── 전체 모델 비교 ───────────────────────────────────────────────────────────

def compare_all_models(
    wrappers: dict,
    test_dataset,
    output_dir: str = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    여러 모델의 성능을 한 번에 평가하고 비교 테이블 반환.

    Args:
        wrappers:     {모델명: ModelWrapper} 딕셔너리
        test_dataset: CASF-2016 테스트 데이터셋
        output_dir:   결과 저장 디렉터리
        batch_size:   배치 크기

    Returns:
        DataFrame 컬럼: [모델명, RMSE, R, R², 학습데이터]

    사용 예시:
        wrappers = {
            "GEMS_CleanSplit": ModelWrapper("GEMS_CleanSplit", gems_models, device),
            "GCN_CleanSplit":  ModelWrapper("GCN_CleanSplit",  [gcn], device, is_ensemble=False),
            "GCN_PDBbind":     ModelWrapper("GCN_PDBbind",     [gcn_pdb], device, is_ensemble=False),
        }
        df = compare_all_models(wrappers, casf2016_dataset, output_dir="results/eval")
    """
    results = []
    for name, wrapper in wrappers.items():
        csv_path = None
        if output_dir:
            csv_path = os.path.join(output_dir, f"{name}_predictions.csv")

        metrics = evaluate_model(wrapper, test_dataset, batch_size, save_csv=csv_path)
        results.append({
            "모델":   name,
            "RMSE":  round(metrics["RMSE"], 4),
            "R":     round(metrics["R"],    4),
            "R²":    round(metrics["R2"],   4),
        })

    df = pd.DataFrame(results).sort_values("RMSE")

    print("\n[Evaluator] ═══ 성능 비교 요약 (CASF-2016) ═══")
    print(df.to_string(index=False))
    print()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_all = os.path.join(output_dir, "comparison_summary.csv")
        df.to_csv(csv_all, index=False)

        json_all = os.path.join(output_dir, "comparison_summary.json")
        with open(json_all, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[Evaluator] 비교 결과 저장: {output_dir}")

    return df


# ─── 빠른 테스트용 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset",  required=True, help="B6AEPL_casf2016.pt 경로")
    parser.add_argument("--gems_model_dir", required=True, help="GEMS/model/ 경로")
    parser.add_argument("--output_dir",    default="results/eval")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.data_loader import load_gems_dataset

    device  = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dataset = load_gems_dataset(args.test_dataset)

    # GEMS 앙상블 로드 (CleanSplit 버전)
    gems_models = auto_load_gems_ensemble(args.gems_model_dir, dataset, device)
    wrapper = ModelWrapper("GEMS_CleanSplit", gems_models, device, is_ensemble=True)

    metrics = evaluate_model(wrapper, dataset, save_csv=os.path.join(args.output_dir, "gems_preds.csv"))
    print("평가 완료:", metrics)
