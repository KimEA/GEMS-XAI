"""
Module 3: Evaluator
====================
모델 성능 평가 모듈.

평가 지표:
    - RMSE (pK 단위, 역정규화 후)
    - Pearson Correlation Coefficient (R)
    - R² (결정계수)

비교 설계 (3×2 팩토리얼):
                     CleanSplit 80%           PDBbind − val_ids
    GC_GNN           GC_GNN_CleanSplit        GC_GNN_PDBbind
    B6AEPL(전체)     GEMS_B6AEPL_CleanSplit   GEMS_B6AEPL_PDBbind
    B6AE0L(리간드)   GEMS_B6AE0L_CleanSplit   GEMS_B6AE0L_PDBbind

테스트셋: B6AEPL_casf2016, B6AEPL_casf2016_indep,
          B6AE0L_casf2016, B6AE0L_casf2016_indep
"""

import os
import csv
import json
import torch
import numpy as np
import pandas as pd

from torch_geometric.loader import DataLoader
from .trainer import load_gems_checkpoint, RMSELoss
from .data_loader import unscale_pk

PK_MIN, PK_MAX = 0.0, 16.0


# ─── 모델 래퍼 ────────────────────────────────────────────────────────────────

class ModelWrapper:
    """
    GEMS18d / GCNNWrapper를 동일한 인터페이스로 래핑.

    Attributes:
        name:        모델 식별 이름
        models:      모델 리스트 (단일 모델이라도 리스트로 관리)
        is_ensemble: True이면 앙상블 평균, False이면 단일 모델
        device:      연산 디바이스
    """
    def __init__(self, name: str, models: list, device, is_ensemble: bool = False):
        self.name        = name
        self.models      = models
        self.is_ensemble = is_ensemble
        self.device      = device

    @torch.no_grad()
    def predict(self, graphbatch) -> torch.Tensor:
        graphbatch = graphbatch.to(self.device)
        if self.is_ensemble:
            preds = [m(graphbatch).view(-1) for m in self.models]
            return torch.mean(torch.stack(preds), dim=0)
        else:
            return self.models[0](graphbatch).view(-1)


# ─── 성능 지표 계산 ───────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, Pearson R, R² 계산."""
    rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r      = float(np.corrcoef(y_true, y_pred)[0, 1])
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
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

    Returns:
        {"RMSE": ..., "R": ..., "R2": ...}
    """
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true_list, y_pred_list, ids = [], [], []
    for batch in loader:
        y_pred_scaled = wrapper.predict(batch).cpu()
        y_true_scaled = batch.y.cpu()

        y_pred_list.append(y_pred_scaled * (PK_MAX - PK_MIN) + PK_MIN)
        y_true_list.append(y_true_scaled * (PK_MAX - PK_MIN) + PK_MIN)
        ids.extend(list(batch.id))

    y_true   = torch.cat(y_true_list).numpy()
    y_pred   = torch.cat(y_pred_list).numpy()
    metrics  = compute_metrics(y_true, y_pred)

    print(f"[Evaluator] {wrapper.name:25s} | "
          f"RMSE={metrics['RMSE']:.4f} | R={metrics['R']:.4f} | R²={metrics['R2']:.4f}")

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
        test_dataset: 테스트 데이터셋
        output_dir:   결과 저장 디렉터리
        batch_size:   배치 크기

    Returns:
        DataFrame 컬럼: [모델명, RMSE, R, R²]
    """
    results = []
    for name, wrapper in wrappers.items():
        csv_path = os.path.join(output_dir, f"{name}_predictions.csv") if output_dir else None
        metrics  = evaluate_model(wrapper, test_dataset, batch_size, save_csv=csv_path)
        results.append({
            "모델": name,
            "RMSE": round(metrics["RMSE"], 4),
            "R":    round(metrics["R"],    4),
            "R²":   round(metrics["R2"],   4),
        })

    df = pd.DataFrame(results).sort_values("RMSE")

    print("\n[Evaluator] ═══ 성능 비교 요약 ═══")
    print(df.to_string(index=False))
    print()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "comparison_summary.csv"), index=False)
        with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return df
