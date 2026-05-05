"""
run_pipeline.py  ─  연구 파이프라인 메인 오케스트레이터
==========================================================
XAI 기반 GEMS 모델의 단백질-리간드 상호작용 학습 메커니즘 검증

연구 질문:
    GEMS가 결합 친화도를 예측할 때 실제로 단백질-리간드 인터페이스(비공유 상호작용
    엣지)에 집중하는가, 아니면 리간드 내부 구조나 훈련 데이터의 통계적 편향에
    의존하는가?

비교 설계:

  [XAI 분석 대상] 4개 모델 × 2개 데이터셋
                     CleanSplit              PDBbind − test_ids
    GC_GNN           GC_GNN_CleanSplit       GC_GNN_PDBbind
    B6AEPL(전체)     GEMS_B6AEPL_CleanSplit  GEMS_B6AEPL_PDBbind

    ─ 열(Column) 비교: 학습 데이터 편향이 XAI 패턴에 미치는 영향
    ─ 행(Row)   비교: 아키텍처 차이 (엣지 특징 사용 여부, GCN vs GEMS)

  [성능 비교 전용] B6AE0L (단백질 노드 제거, 음성 대조군)
    GEMS_B6AE0L_CleanSplit, GEMS_B6AE0L_PDBbind
    → B6AE0L 그래프는 interaction 엣지가 구조상 0개이므로 XAI 불가
    → CASF-2016 RMSE/R로 "단백질 맥락 제거 시 성능 저하" 확인

데이터 분할 전략 (80:10:10):
    ┌─────────────────────────────────────────────────────────┐
    │  Train 80%  │  Val 10% (Early stopping)  │  Test 10%   │
    │             │  ← 모델이 "보는" 데이터 →  │  ← XAI 분석│
    │             │                             │    홀드아웃 │
    └─────────────────────────────────────────────────────────┘
    - XAI 홀드아웃(test): 학습/early stopping 어디에도 사용되지 않는 완전 독립 셋
    - 16,491 × 10% ≈ 1,649개 → Low/Medium/High 각 ~500개 이상으로 통계 검정 충분

성능 평가 테스트셋 (CASF-2016, 별도):
    - B6AEPL_casf2016        : CASF-2016 전체 (GC_GNN + GEMS_B6AEPL 평가)
    - B6AEPL_casf2016_indep  : CASF-2016 독립 부분집합
    - B6AE0L_casf2016        : CASF-2016 전체 (단백질 제거)
    - B6AE0L_casf2016_indep  : CASF-2016 독립 부분집합

실행 단계:
    Step  1. B6AEPL CleanSplit 로드 + ID 기반 80:10:10 3분할 (JSON 저장)
    Step  2. B6AE0L CleanSplit 로드 + 동일 3분할 ID 적용
    Step  3. PDBbind B6AEPL + B6AE0L 로드 + val/test ID 제거 (누수 방지)
    Step  4. GEMS_B6AEPL_CleanSplit 학습
    Step  5. GEMS_B6AE0L_CleanSplit 학습
    Step  6. GEMS_B6AEPL_PDBbind 학습
    Step  7. GEMS_B6AE0L_PDBbind 학습
    Step  8. GC_GNN_CleanSplit 학습
    Step  9. GC_GNN_PDBbind 학습
    Step 10. CASF-2016 + CASF-2016_indep 성능 평가 (RMSE, R)
    Step 11. EdgeSHAPer XAI 분석 (XAI 홀드아웃 10%)
    Step 12. 시각화

사용 예시:
    KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/pli_m1/bin/python \\
        03_analysis/ver2/run_pipeline.py \\
        --cleansplit_b6aepl      02_data/GEMS_pytorch_datasets/B6AEPL_train_cleansplit.pt \\
        --cleansplit_b6ae0l      02_data/GEMS_pytorch_datasets/B6AE0L_train_cleansplit.pt \\
        --pdbbind_b6aepl         02_data/GEMS_pytorch_datasets/B6AEPL_train_pdbbind.pt \\
        --pdbbind_b6ae0l         02_data/GEMS_pytorch_datasets/B6AE0L_train_pdbbind.pt \\
        --casf2016_b6aepl        02_data/GEMS_pytorch_datasets/B6AEPL_casf2016.pt \\
        --casf2016_b6aepl_indep  02_data/GEMS_pytorch_datasets/B6AEPL_casf2016_indep.pt \\
        --casf2016_b6ae0l        02_data/GEMS_pytorch_datasets/B6AE0L_casf2016.pt \\
        --casf2016_b6ae0l_indep  02_data/GEMS_pytorch_datasets/B6AE0L_casf2016_indep.pt \\
        --output_dir             03_analysis/ver2/results/pipeline \\
        --M 50 --max_per_group 200

서버 실행 시 변경 사항:
    - KMP_DUPLICATE_LIB_OK=TRUE 환경 변수 불필요 (Linux)
    - device는 자동으로 cuda로 선택됨
    - DataLoader num_workers를 4 이상으로 늘려도 됨
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from functools import partial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "GEMS"))

DATA_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "02_data", "GEMS_pytorch_datasets")

# ─── 파이프라인 모듈 임포트 ───────────────────────────────────────────────────
from pipeline.data_loader import (
    load_gems_dataset, get_dataset_info,
    split_train_val_test, load_id_split, apply_id_split, exclude_ids,
    stratify_by_affinity, create_dataloader,
)
from pipeline.trainer import (
    train_model,
    build_gems18d, save_gems_checkpoint, load_gems_checkpoint,
    GCNNWrapper, build_gcngnn, save_gcngnn_checkpoint, load_gcngnn_checkpoint,
)
from pipeline.evaluator import (
    ModelWrapper, evaluate_model, compare_all_models,
)
from pipeline.xai_analyzer import (
    run_full_xai_analysis, DEFAULT_K_VALUES,
)
from pipeline.visualizer import (
    plot_performance_comparison, plot_edge_barchart,
    plot_topk_lineplot, plot_multi_model_comparison,
)


# ─── 인자 파싱 ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GEMS XAI 파이프라인 — 3×2 팩토리얼 설계",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 학습 데이터
    p.add_argument("--cleansplit_b6aepl",
        default=os.path.join(DATA_ROOT, "B6AEPL_train_cleansplit.pt"))
    p.add_argument("--cleansplit_b6ae0l",
        default=os.path.join(DATA_ROOT, "B6AE0L_train_cleansplit.pt"))
    p.add_argument("--pdbbind_b6aepl",
        default=os.path.join(DATA_ROOT, "B6AEPL_train_pdbbind.pt"))
    p.add_argument("--pdbbind_b6ae0l",
        default=os.path.join(DATA_ROOT, "B6AE0L_train_pdbbind.pt"))

    # ── 테스트 데이터 (표준 + 독립 부분집합)
    p.add_argument("--casf2016_b6aepl",
        default=os.path.join(DATA_ROOT, "B6AEPL_casf2016.pt"))
    p.add_argument("--casf2016_b6aepl_indep",
        default=os.path.join(DATA_ROOT, "B6AEPL_casf2016_indep.pt"))
    p.add_argument("--casf2016_b6ae0l",
        default=os.path.join(DATA_ROOT, "B6AE0L_casf2016.pt"))
    p.add_argument("--casf2016_b6ae0l_indep",
        default=os.path.join(DATA_ROOT, "B6AE0L_casf2016_indep.pt"))

    # ── ID 분할 파일
    p.add_argument("--split_json", default=None,
        help="기존 ID 분할 JSON. 없으면 output_dir/id_split.json 자동 생성")

    # ── 결과 저장
    p.add_argument("--output_dir",
        default=os.path.join(SCRIPT_DIR, "results", "pipeline"))

    # ── 학습 제어
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_eval",  action="store_true")
    p.add_argument("--skip_xai",   action="store_true")

    # ── GEMS 체크포인트 (--skip_train 시 사용)
    p.add_argument("--gems_b6aepl_cleansplit_ckpt", default=None)
    p.add_argument("--gems_b6ae0l_cleansplit_ckpt", default=None)
    p.add_argument("--gems_b6aepl_pdbbind_ckpt",   default=None)
    p.add_argument("--gems_b6ae0l_pdbbind_ckpt",   default=None)

    # ── GC_GNN 체크포인트 (--skip_train 시 사용)
    p.add_argument("--gcn_cleansplit_ckpt", default=None)
    p.add_argument("--gcn_pdbbind_ckpt",   default=None)

    # ── GEMS 학습 하이퍼파라미터
    # 원논문: SGD(momentum=0.9), lr=1e-3, wd=5e-4, bs=32, epochs=2000, dropout=0.5
    # nesterov: 원논문 미명시(False). scheduler: 원논문 기본 False
    # 본 연구: epochs=2000 + early stopping(patience=50) 추가 — 논문에 명시
    p.add_argument("--gems_lr",           type=float, default=1e-3)
    p.add_argument("--gems_epochs",       type=int,   default=2000)
    p.add_argument("--gems_patience",     type=int,   default=100)
    p.add_argument("--gems_es_tol",       type=float, default=0.01,
                   help="GEMS early stopping 최소 개선량 (원논문: 0.01)")
    p.add_argument("--gems_batch_size",   type=int,   default=32)
    p.add_argument("--gems_weight_decay", type=float, default=5e-4)
    p.add_argument("--gems_optimizer",    type=str,   default="sgd",
                   choices=["adam", "sgd"],
                   help="GEMS 원논문: SGD(momentum=0.9)")
    p.add_argument("--gems_momentum",     type=float, default=0.9,
                   help="SGD momentum (gems_optimizer=sgd 시 사용)")
    p.add_argument("--gems_scheduler",    action="store_true", default=False,
                   help="LR 스케줄러 사용 여부 (원논문 기본 False)")
    p.add_argument("--gems_dropout",      type=float, default=0.5)

    # ── GC_GNN 학습 하이퍼파라미터
    # 원논문: Adam, lr=1e-4, wd=5e-4, bs=32, hidden=256
    p.add_argument("--gcn_lr",           type=float, default=1e-4)
    p.add_argument("--gcn_epochs",       type=int,   default=100)
    p.add_argument("--gcn_patience",     type=int,   default=50)
    p.add_argument("--gcn_batch_size",   type=int,   default=32)
    p.add_argument("--gcn_weight_decay", type=float, default=5e-4)
    p.add_argument("--gcn_optimizer",    type=str,   default="adam",
                   choices=["adam", "sgd"],
                   help="GC_GNN 원논문: Adam")
    p.add_argument("--gcn_hidden",       type=int,   default=256)

    # ── XAI 설정
    p.add_argument("--M",             type=int,   default=100,
                   help="EdgeSHAPer Monte Carlo 반복 수 (원논문: 100)")
    p.add_argument("--max_per_group", type=int,   default=None)
    p.add_argument("--train_ratio",   type=float, default=0.8)
    p.add_argument("--val_ratio",     type=float, default=0.1)
    p.add_argument("--seed",          type=int,   default=42,
                   help="데이터 분할 시드 (고정). 모델 학습 시드는 --seeds로 별도 지정")
    p.add_argument("--seeds",         type=str,   default="42,123,2024",
                   help="모델 학습 시드 목록 (쉼표 구분). 결과는 평균±표준편차로 보고")
    p.add_argument("--num_workers",   type=int,   default=0,
                   help="DataLoader 병렬 워커 수 (CPU/MPS: 0, CUDA 서버: 4~8 권장)")

    return p.parse_args()


# ─── 헬퍼 ────────────────────────────────────────────────────────────────────

def mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _train_gems(train_data, val_data, info, config, ckpt_dir, name, device, seed: int = 42, num_workers: int = 0):
    """GEMS18d 인스턴스화 → 학습 → 체크포인트 저장.

    원논문 하이퍼파라미터: SGD(momentum=0.9), lr=1e-3, wd=5e-4, bs=32, epochs=2000
    본 연구: 동일 설정 + early stopping(patience=100, tol=0.01) 추가 — 과적합 방지 목적으로 논문에 명시
    """
    torch.manual_seed(seed)
    model = build_gems18d(
        node_feat_dim     = info["node_feat_dim"],
        edge_feat_dim     = info["edge_feat_dim"],
        dropout_prob      = config["gems_dropout"],
        conv_dropout_prob = 0,
        device            = device,
    )
    pin = (device.type == "cuda")
    train_loader = create_dataloader(train_data, config["gems_batch_size"], shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = create_dataloader(val_data,   config["gems_batch_size"], shuffle=False, num_workers=num_workers, pin_memory=pin)

    gems_save = partial(
        save_gems_checkpoint,
        node_feat_dim = info["node_feat_dim"],
        edge_feat_dim = info["edge_feat_dim"],
    )
    train_config = {
        "lr":           config["gems_lr"],
        "weight_decay": config["gems_weight_decay"],
        "optimizer":    config["gems_optimizer"],
        "momentum":     config["gems_momentum"],
        "epochs":       config["gems_epochs"],
        "patience":     config["gems_patience"],
        "es_tolerance": config["gems_es_tol"],
        "scheduler":    config["gems_scheduler"],
        "seed":         seed,
    }
    train_model(model, train_loader, val_loader, train_config, device,
                save_dir=ckpt_dir, model_name=name, save_fn=gems_save)
    return model


def _train_gcn(train_data, val_data, info, config, ckpt_dir, name, device, seed: int = 42, num_workers: int = 0):
    """논문 원본 GC_GNN(GCNNWrapper) 인스턴스화 → 학습 → 체크포인트 저장.

    원논문 하이퍼파라미터: Adam, lr=1e-4, weight_decay=5e-4, bs=32, hidden=256, epochs=100
    """
    torch.manual_seed(seed)
    model = build_gcngnn(
        node_feat_dim = info["node_feat_dim"],
        hidden        = config["gcn_hidden"],
        device        = device,
    )
    pin = (device.type == "cuda")
    train_loader = create_dataloader(train_data, config["gcn_batch_size"], shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = create_dataloader(val_data,   config["gcn_batch_size"], shuffle=False, num_workers=num_workers, pin_memory=pin)

    gcn_save = partial(
        save_gcngnn_checkpoint,
        node_feat_dim = info["node_feat_dim"],
        hidden        = config["gcn_hidden"],
    )
    train_config = {
        "lr":           config["gcn_lr"],
        "weight_decay": config["gcn_weight_decay"],
        "optimizer":    config["gcn_optimizer"],
        "epochs":       config["gcn_epochs"],
        "patience":     config["gcn_patience"],
        "scheduler":    False,
        "seed":         seed,
    }
    train_model(model, train_loader, val_loader, train_config, device,
                save_dir=ckpt_dir, model_name=name, save_fn=gcn_save)
    return model


def _make_wrapper(name, models, device):
    """모델 리스트 → ModelWrapper. 2개 이상이면 앙상블."""
    return ModelWrapper(name, models, device, is_ensemble=len(models) > 1)


def _aggregate_xai_across_seeds(xai_per_seed: list, k_values: list) -> dict:
    """
    여러 시드 XAI 결과를 샘플별로 평균.

    각 (모델, 그룹, 샘플_i) 에 대해 topk_stats[k][field]를 시드 평균.
    '__stat_tests__' 키는 자동 제외.
    동일 그룹의 샘플 순서가 모든 시드에서 동일하다고 가정
    (max_samples 서브샘플링 시드를 고정하면 보장됨).
    """
    if not xai_per_seed:
        return {}

    first    = xai_per_seed[0]
    n_seeds  = len(xai_per_seed)
    aggregated = {}

    for model_name in first:
        if model_name == "__stat_tests__":
            continue
        aggregated[model_name] = {}

        for group_name in first[model_name]:
            # 시드별 per_sample 리스트 수집
            per_seed_samples = [
                xai_per_seed[si][model_name][group_name]["per_sample"]
                for si in range(n_seeds)
                if (model_name in xai_per_seed[si]
                    and group_name in xai_per_seed[si][model_name])
            ]
            if not per_seed_samples:
                continue

            n_seeds_actual = len(per_seed_samples)   # 실제 사용된 시드 수
            if n_seeds_actual < n_seeds:
                print(f"[경고] {model_name}/{group_name}: "
                      f"{n_seeds}개 시드 중 {n_seeds_actual}개만 결과 존재 — "
                      f"{n_seeds_actual}개로 평균합니다.")

            lengths = [len(s) for s in per_seed_samples]
            if len(set(lengths)) > 1:
                print(f"[경고] {model_name}/{group_name}: 시드별 샘플 수 불일치 "
                      f"{lengths} — 최소값({min(lengths)})으로 잘라서 평균합니다. "
                      f"XAI 서브샘플링 시드가 시드별로 다를 수 있습니다.")
            n_samples = min(lengths)
            averaged_samples = []

            for i in range(n_samples):
                base = {k: v for k, v in per_seed_samples[0][i].items()
                        if k != "topk_stats"}
                avg_topk = {}
                for k in k_values:
                    avg_topk[k] = {}
                    for field in per_seed_samples[0][i]["topk_stats"][k]:
                        vals = [
                            per_seed_samples[si][i]["topk_stats"][k][field]
                            for si in range(len(per_seed_samples))
                        ]
                        finite = [v for v in vals
                                  if isinstance(v, (int, float)) and not np.isnan(v)]
                        avg_topk[k][field] = float(np.mean(finite)) if finite else float("nan")
                base["topk_stats"] = avg_topk
                averaged_samples.append(base)

            # mean_topk 재계산
            mean_topk = {}
            for k in k_values:
                def _mn(field, samps=averaged_samples, _k=k):
                    vals = [s["topk_stats"][_k][field] for s in samps]
                    finite = [v for v in vals if not np.isnan(v)]
                    return float(np.mean(finite)) if finite else float("nan")

                lig_r  = [s["topk_stats"][k]["ligand"]     for s in averaged_samples]
                int_r  = [s["topk_stats"][k]["interaction"] for s in averaged_samples]
                prot_r = [s["topk_stats"][k]["protein"]     for s in averaged_samples]
                lift_i = [s["topk_stats"][k].get("lift_interaction", float("nan"))
                          for s in averaged_samples]
                finite_lift = [v for v in lift_i if not np.isnan(v)]

                mean_topk[k] = {
                    "ligand":               float(np.mean(lig_r)),
                    "interaction":          float(np.mean(int_r)),
                    "protein":              float(np.mean(prot_r)),
                    "std_ligand":           float(np.std(lig_r)),
                    "std_interaction":      float(np.std(int_r)),
                    "baseline_interaction": _mn("baseline_interaction"),
                    "lift_ligand":          _mn("lift_ligand"),
                    "lift_interaction":     _mn("lift_interaction"),
                    "lift_protein":         _mn("lift_protein"),
                    "std_lift_within_testset": float(np.std(finite_lift)) if finite_lift else 0.0,
                }

            aggregated[model_name][group_name] = {
                "group":      group_name,
                "model":      model_name,
                "n_samples":  n_samples,
                "n_seeds":    n_seeds_actual,
                "per_sample": averaged_samples,
                "mean_topk":  mean_topk,
            }

    return aggregated


def _report_seed_metrics(per_seed_metrics: list, model_name: str, seeds: list) -> dict:
    """
    시드별 평가 지표 리스트 → mean±std 요약 출력 및 딕셔너리 반환.
    per_seed_metrics: [{"RMSE": ..., "R": ..., "R2": ...}, ...]
    """
    for seed, m in zip(seeds, per_seed_metrics):
        print(f"    seed={seed}: RMSE={m['RMSE']:.4f} R={m['R']:.4f} R²={m['R2']:.4f}")
    mean_m = {k: float(np.mean([m[k] for m in per_seed_metrics])) for k in ["RMSE", "R", "R2"]}
    std_m  = {k: float(np.std( [m[k] for m in per_seed_metrics])) for k in ["RMSE", "R", "R2"]}
    print(f"  → {model_name}: "
          f"RMSE={mean_m['RMSE']:.4f}±{std_m['RMSE']:.4f}  "
          f"R={mean_m['R']:.4f}±{std_m['R']:.4f}  "
          f"R²={mean_m['R2']:.4f}±{std_m['R2']:.4f}")
    return {"mean": mean_m, "std": std_m}


def _try_load(path):
    """파일이 존재하면 로드, 없으면 None."""
    if path and os.path.exists(path):
        return load_gems_dataset(path)
    if path:
        print(f"[경고] 파일 없음, 건너뜀: {path}")
    return None


def _eval_on_datasets(wrappers, test_datasets, eval_subdir, batch_size):
    """
    wrappers: {model_name: ModelWrapper}
    test_datasets: {"casf2016": ds, "casf2016_indep": ds_indep}
    각 테스트셋에서 compare_all_models 실행 후 결과를 통합하여 반환.
    """
    all_metrics = {}
    for ds_label, dataset in test_datasets.items():
        if dataset is None:
            continue
        df = compare_all_models(wrappers, dataset,
                                os.path.join(eval_subdir, ds_label), batch_size)
        for _, row in df.iterrows():
            key = f"{row['모델']} [{ds_label}]"
            all_metrics[key] = {"RMSE": row["RMSE"], "R": row["R"], "R²": row["R²"]}
    return all_metrics


# ─── 메인 파이프라인 ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t_start = time.time()

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  GEMS XAI 파이프라인  (3×2 팩토리얼, 6개 모델)")
    print(f"  디바이스: {device}")
    print(f"  출력 디렉터리: {args.output_dir}")
    print(f"{'='*60}\n")

    ckpt_dir = mkdir(os.path.join(args.output_dir, "checkpoints"))
    eval_dir = mkdir(os.path.join(args.output_dir, "evaluation"))
    xai_dir  = mkdir(os.path.join(args.output_dir, "xai"))
    viz_dir  = mkdir(os.path.join(args.output_dir, "figures"))

    gems_config = {
        "gems_lr":           args.gems_lr,
        "gems_epochs":       args.gems_epochs,
        "gems_patience":     args.gems_patience,
        "gems_es_tol":       args.gems_es_tol,
        "gems_batch_size":   args.gems_batch_size,
        "gems_weight_decay": args.gems_weight_decay,
        "gems_optimizer":    args.gems_optimizer,
        "gems_momentum":     args.gems_momentum,
        "gems_scheduler":    args.gems_scheduler,
        "gems_dropout":      args.gems_dropout,
    }
    gcn_config = {
        "gcn_lr":           args.gcn_lr,
        "gcn_epochs":       args.gcn_epochs,
        "gcn_patience":     args.gcn_patience,
        "gcn_batch_size":   args.gcn_batch_size,
        "gcn_weight_decay": args.gcn_weight_decay,
        "gcn_optimizer":    args.gcn_optimizer,
        "gcn_hidden":       args.gcn_hidden,
    }

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"  학습 시드: {seeds} (결과는 mean±std로 보고)")

    # ────────────────────────────────────────────────────────────────
    # Step 1: B6AEPL CleanSplit + ID 기반 80:10:10 3분할
    # ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Step 1. B6AEPL CleanSplit 로드 + 80:10:10 3분할")
    print(f"{'─'*50}")

    b6aepl_clean = load_gems_dataset(args.cleansplit_b6aepl)
    info_aepl    = get_dataset_info(b6aepl_clean)

    split_json = args.split_json or os.path.join(args.output_dir, "id_split.json")
    if os.path.exists(split_json):
        print(f"[Step 1] 기존 ID 분할 재사용: {split_json}")
        train_ids, val_ids, test_ids = load_id_split(split_json)
        train_b6aepl, val_b6aepl, test_b6aepl = apply_id_split(
            b6aepl_clean, train_ids, val_ids, test_ids)
    else:
        train_b6aepl, val_b6aepl, test_b6aepl, train_ids, val_ids, test_ids = \
            split_train_val_test(
                b6aepl_clean,
                train_ratio = args.train_ratio,
                val_ratio   = args.val_ratio,
                seed        = args.seed,
                save_path   = split_json,
            )

    print(f"\n[Step 1] XAI 홀드아웃 층화 (B6AEPL 10%)...")
    group_indices_aepl = stratify_by_affinity(test_b6aepl)
    with open(os.path.join(args.output_dir, "stratification_b6aepl.json"), "w") as f:
        json.dump({g: len(idx) for g, idx in group_indices_aepl.items()}, f, indent=2)

    # ────────────────────────────────────────────────────────────────
    # Step 2: B6AE0L CleanSplit + 동일 3분할 ID 적용
    # ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Step 2. B6AE0L CleanSplit 로드 + 동일 3분할 적용")
    print(f"{'─'*50}")

    b6ae0l_clean = load_gems_dataset(args.cleansplit_b6ae0l)
    info_ae0l    = get_dataset_info(b6ae0l_clean)
    train_b6ae0l, val_b6ae0l, test_b6ae0l = apply_id_split(
        b6ae0l_clean, train_ids, val_ids, test_ids)

    # B6AE0L은 XAI 불가 (interaction 엣지 0개) → 성능 평가 전용이므로 층화 불필요

    # ────────────────────────────────────────────────────────────────
    # Step 3: PDBbind 로드 + val/test ID 제거 (누수 방지)
    # ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Step 3. PDBbind 로드 + val/test ID 누수 방지")
    print(f"{'─'*50}")

    # val과 test 모두 PDBbind 학습셋에서 제거해야 누수 없음
    exclude_from_pdbbind = val_ids | test_ids
    pdbbind_aepl_raw   = load_gems_dataset(args.pdbbind_b6aepl)
    pdbbind_ae0l_raw   = load_gems_dataset(args.pdbbind_b6ae0l)
    pdbbind_aepl_clean = exclude_ids(pdbbind_aepl_raw, exclude_from_pdbbind)
    pdbbind_ae0l_clean = exclude_ids(pdbbind_ae0l_raw, exclude_from_pdbbind)

    # PDBbind early stopping: val_b6aepl/val_b6ae0l (10%)를 그대로 사용
    # → CleanSplit val과 동일 ID이므로 비율·조건이 일치함
    val_pdb_aepl_tmp = val_b6aepl
    val_pdb_ae0l_tmp = val_b6ae0l

    # ────────────────────────────────────────────────────────────────
    # Step 4-9: 모델 학습 (GEMS 4개 + GC_GNN 2개) × N seeds
    # ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Step 4-9. 모델 학습 ({len(seeds)}개 시드 × 6개 모델)")
    print(f"{'─'*50}")

    # 모델 타입별 시드별 모델 리스트
    gems_b6aepl_clean_models   = []
    gems_b6ae0l_clean_models   = []
    gems_b6aepl_pdbbind_models = []
    gems_b6ae0l_pdbbind_models = []
    gcn_clean_models           = []
    gcn_pdbbind_models         = []

    if not args.skip_train:
        for seed in seeds:
            print(f"\n{'━'*40}")
            print(f"  시드 {seed} 학습 시작")
            print(f"{'━'*40}")

            print(f"\n[Step 4|seed={seed}] GEMS_B6AEPL_CleanSplit...")
            gems_b6aepl_clean_models.append(_train_gems(
                train_b6aepl, val_b6aepl, info_aepl, gems_config,
                ckpt_dir, f"gems_b6aepl_cleansplit_seed{seed}", device, seed=seed,
                num_workers=args.num_workers,
            ))

            print(f"\n[Step 5|seed={seed}] GEMS_B6AE0L_CleanSplit...")
            gems_b6ae0l_clean_models.append(_train_gems(
                train_b6ae0l, val_b6ae0l, info_ae0l, gems_config,
                ckpt_dir, f"gems_b6ae0l_cleansplit_seed{seed}", device, seed=seed,
                num_workers=args.num_workers,
            ))

            print(f"\n[Step 6|seed={seed}] GEMS_B6AEPL_PDBbind...")
            gems_b6aepl_pdbbind_models.append(_train_gems(
                pdbbind_aepl_clean, val_pdb_aepl_tmp, info_aepl, gems_config,
                ckpt_dir, f"gems_b6aepl_pdbbind_seed{seed}", device, seed=seed,
                num_workers=args.num_workers,
            ))

            print(f"\n[Step 7|seed={seed}] GEMS_B6AE0L_PDBbind...")
            gems_b6ae0l_pdbbind_models.append(_train_gems(
                pdbbind_ae0l_clean, val_pdb_ae0l_tmp, info_ae0l, gems_config,
                ckpt_dir, f"gems_b6ae0l_pdbbind_seed{seed}", device, seed=seed,
                num_workers=args.num_workers,
            ))

            print(f"\n[Step 8|seed={seed}] GC_GNN_CleanSplit...")
            gcn_clean_models.append(_train_gcn(
                train_b6aepl, val_b6aepl, info_aepl, gcn_config,
                ckpt_dir, f"gcn_cleansplit_seed{seed}", device, seed=seed,
                num_workers=args.num_workers,
            ))

            print(f"\n[Step 9|seed={seed}] GC_GNN_PDBbind...")
            gcn_pdbbind_models.append(_train_gcn(
                pdbbind_aepl_clean, val_pdb_aepl_tmp, info_aepl, gcn_config,
                ckpt_dir, f"gcn_pdbbind_seed{seed}", device, seed=seed,
                num_workers=args.num_workers,
            ))

    else:
        # skip_train: 체크포인트 디렉터리에서 seed별 파일 자동 탐색
        # 없으면 --*_ckpt 인자에서 단일 모델 로드 (seed=1개로 처리)
        def _load_seed_ckpts(base_name, load_fn):
            models = []
            for seed in seeds:
                path = os.path.join(ckpt_dir, f"{base_name}_seed{seed}_best.pt")
                if os.path.exists(path):
                    models.append(load_fn(path, device))
                    print(f"[로드] {base_name} seed={seed}: {path}")
            return models

        gems_b6aepl_clean_models   = _load_seed_ckpts("gems_b6aepl_cleansplit", load_gems_checkpoint)
        gems_b6ae0l_clean_models   = _load_seed_ckpts("gems_b6ae0l_cleansplit", load_gems_checkpoint)
        gems_b6aepl_pdbbind_models = _load_seed_ckpts("gems_b6aepl_pdbbind",    load_gems_checkpoint)
        gems_b6ae0l_pdbbind_models = _load_seed_ckpts("gems_b6ae0l_pdbbind",    load_gems_checkpoint)
        gcn_clean_models           = _load_seed_ckpts("gcn_cleansplit",          load_gcngnn_checkpoint)
        gcn_pdbbind_models         = _load_seed_ckpts("gcn_pdbbind",             load_gcngnn_checkpoint)

        # fallback: 기존 단일 체크포인트 인자
        if not gems_b6aepl_clean_models and args.gems_b6aepl_cleansplit_ckpt:
            gems_b6aepl_clean_models = [load_gems_checkpoint(args.gems_b6aepl_cleansplit_ckpt, device)]
        if not gems_b6ae0l_clean_models and args.gems_b6ae0l_cleansplit_ckpt:
            gems_b6ae0l_clean_models = [load_gems_checkpoint(args.gems_b6ae0l_cleansplit_ckpt, device)]
        if not gems_b6aepl_pdbbind_models and args.gems_b6aepl_pdbbind_ckpt:
            gems_b6aepl_pdbbind_models = [load_gems_checkpoint(args.gems_b6aepl_pdbbind_ckpt, device)]
        if not gems_b6ae0l_pdbbind_models and args.gems_b6ae0l_pdbbind_ckpt:
            gems_b6ae0l_pdbbind_models = [load_gems_checkpoint(args.gems_b6ae0l_pdbbind_ckpt, device)]
        if not gcn_clean_models and args.gcn_cleansplit_ckpt:
            gcn_clean_models = [load_gcngnn_checkpoint(args.gcn_cleansplit_ckpt, device)]
        if not gcn_pdbbind_models and args.gcn_pdbbind_ckpt:
            gcn_pdbbind_models = [load_gcngnn_checkpoint(args.gcn_pdbbind_ckpt, device)]

    for models in [gems_b6aepl_clean_models, gems_b6ae0l_clean_models,
                   gems_b6aepl_pdbbind_models, gems_b6ae0l_pdbbind_models,
                   gcn_clean_models, gcn_pdbbind_models]:
        for m in models:
            m.eval()

    # ────────────────────────────────────────────────────────────────
    # CASF-2016 데이터셋 로드 (평가 + XAI 공용)
    # ────────────────────────────────────────────────────────────────
    casf_aepl      = _try_load(args.casf2016_b6aepl)
    casf_aepl_indep= _try_load(args.casf2016_b6aepl_indep)
    casf_ae0l      = _try_load(args.casf2016_b6ae0l)
    casf_ae0l_indep= _try_load(args.casf2016_b6ae0l_indep)

    # CASF-2016 XAI용 층화: B6AEPL만 (B6AE0L은 interaction 엣지 0개 → XAI 불가)
    group_indices_casf_aepl = stratify_by_affinity(casf_aepl) if (casf_aepl and not args.skip_xai) else {}
    if group_indices_casf_aepl:
        with open(os.path.join(args.output_dir, "stratification_casf2016_b6aepl.json"), "w") as f:
            json.dump({g: len(idx) for g, idx in group_indices_casf_aepl.items()}, f, indent=2)

    # ────────────────────────────────────────────────────────────────
    # Step 10: CASF-2016 + CASF-2016_indep 평가 (per-seed + 앙상블)
    # ────────────────────────────────────────────────────────────────
    if not args.skip_eval:
        print(f"\n{'─'*50}")
        print("Step 10. CASF-2016 성능 평가 — per-seed mean±std + 앙상블")
        print(f"{'─'*50}")

        from pipeline.evaluator import evaluate_model as _eval_model

        def _eval_model_seeds(models_list, display_name, test_ds):
            """시드별 개별 평가 후 mean±std 출력. 앙상블 wrapper 반환."""
            if not models_list or test_ds is None:
                return None, {}
            eval_seeds = seeds[:len(models_list)]
            if len(models_list) != len(seeds):
                print(f"[경고] {display_name}: seeds={len(seeds)}개, "
                      f"모델={len(models_list)}개 불일치 — "
                      f"{len(eval_seeds)}개 seed로 평가합니다.")
            per_seed_m = []
            for seed, m in zip(eval_seeds, models_list):
                w = ModelWrapper(f"{display_name}_seed{seed}", [m], device)
                per_seed_m.append(_eval_model(w, test_ds, batch_size=256))
            summary = _report_seed_metrics(per_seed_m, display_name, eval_seeds)
            ens = _make_wrapper(display_name, models_list, device)
            ens_metrics = _eval_model(ens, test_ds, batch_size=256)
            return ens, {
                "per_seed": per_seed_m,
                "mean":     summary["mean"],
                "std":      summary["std"],
                "ensemble": ens_metrics,
            }

        all_eval_summary = {}
        ensemble_wrappers_aepl = {}
        ensemble_wrappers_ae0l = {}

        for ds_label, casf_ds in [("casf2016", casf_aepl), ("casf2016_indep", casf_aepl_indep)]:
            if casf_ds is None:
                continue
            print(f"\n  ── {ds_label} (B6AEPL 그래프) ──")
            for name, mlist in [
                ("GEMS_B6AEPL_CleanSplit", gems_b6aepl_clean_models),
                ("GEMS_B6AEPL_PDBbind",   gems_b6aepl_pdbbind_models),
                ("GC_GNN_CleanSplit",      gcn_clean_models),
                ("GC_GNN_PDBbind",         gcn_pdbbind_models),
            ]:
                ens_w, summary = _eval_model_seeds(mlist, name, casf_ds)
                if ens_w is not None:
                    all_eval_summary[f"{name} [{ds_label}]"] = summary
                    if ds_label == "casf2016":
                        ensemble_wrappers_aepl[name] = ens_w

        for ds_label, casf_ds in [("casf2016", casf_ae0l), ("casf2016_indep", casf_ae0l_indep)]:
            if casf_ds is None:
                continue
            print(f"\n  ── {ds_label} (B6AE0L 그래프) ──")
            for name, mlist in [
                ("GEMS_B6AE0L_CleanSplit", gems_b6ae0l_clean_models),
                ("GEMS_B6AE0L_PDBbind",   gems_b6ae0l_pdbbind_models),
            ]:
                ens_w, summary = _eval_model_seeds(mlist, name, casf_ds)
                if ens_w is not None:
                    all_eval_summary[f"{name} [{ds_label}]"] = summary
                    if ds_label == "casf2016":
                        ensemble_wrappers_ae0l[name] = ens_w

        with open(os.path.join(eval_dir, "eval_seed_summary.json"), "w") as f:
            json.dump(all_eval_summary, f, indent=2, ensure_ascii=False)

        # 앙상블 성능으로 시각화
        ens_metrics_for_plot = {
            k: v["ensemble"] for k, v in all_eval_summary.items()
        }
        if ens_metrics_for_plot:
            plot_performance_comparison(
                ens_metrics_for_plot,
                save_path=os.path.join(viz_dir, "performance_comparison.png"),
            )

    # ────────────────────────────────────────────────────────────────
    # Step 11: XAI 분석 — 시드별 실행 후 결과 평균
    # ────────────────────────────────────────────────────────────────
    if not args.skip_xai:
        print(f"\n{'─'*50}")
        print(f"Step 11. XAI 분석 (EdgeSHAPer, 4모델 × {len(seeds)}시드, B6AEPL 전용)")
        print(f"{'─'*50}")
        print(f"  ※ XAI 대상: 홀드아웃 {len(test_b6aepl)}개 + CASF-2016 {len(casf_aepl) if casf_aepl else 0}개")
        print(f"  ※ B6AE0L은 interaction 엣지 0개 → 성능 비교 전용(Step 10)")
        print(f"  ※ 각 시드별 단일 모델로 XAI 수행 후 topk_stats 평균")

        from pipeline.xai_analyzer import compare_groups_statistically

        # XAI 모델 타입별 시드-인덱스 대응
        xai_model_seed_map = {
            "GEMS_B6AEPL_CleanSplit": gems_b6aepl_clean_models,
            "GEMS_B6AEPL_PDBbind":   gems_b6aepl_pdbbind_models,
            "GC_GNN_CleanSplit":      gcn_clean_models,
            "GC_GNN_PDBbind":         gcn_pdbbind_models,
        }
        # 모델이 없는 항목 제거
        xai_model_seed_map = {k: v for k, v in xai_model_seed_map.items() if v}

        def _run_xai_multiset(val_dataset, group_indices, xai_subdir, label):
            """시드별 XAI 실행 → 집계 → 통계 검정."""
            if not xai_model_seed_map or not group_indices:
                return {}

            xai_per_seed = []
            n_s = min(len(v) for v in xai_model_seed_map.values())

            for si, seed in enumerate(seeds[:n_s]):
                print(f"\n[{label}] 시드 {seed} ({si+1}/{n_s}) XAI 실행...")
                seed_wrappers = {
                    name: ModelWrapper(name, [mlist[si]], device, is_ensemble=False)
                    for name, mlist in xai_model_seed_map.items()
                }
                result = run_full_xai_analysis(
                    wrappers      = seed_wrappers,
                    val_dataset   = val_dataset,
                    group_indices = group_indices,
                    M             = args.M,
                    k_values      = DEFAULT_K_VALUES,
                    output_dir    = mkdir(os.path.join(xai_subdir, f"seed{seed}")),
                    max_per_group = args.max_per_group,
                    device        = device,
                )
                result.pop("__stat_tests__", None)
                xai_per_seed.append(result)

            print(f"\n[{label}] {n_s}개 시드 결과 집계...")
            aggregated = _aggregate_xai_across_seeds(xai_per_seed, DEFAULT_K_VALUES)

            # 집계 결과 저장
            agg_path = os.path.join(xai_subdir, "aggregated_mean_topk.json")
            agg_save = {
                mn: {
                    gn: {"n_samples": gs["n_samples"], "n_seeds": gs["n_seeds"],
                         "mean_topk": gs["mean_topk"]}
                    for gn, gs in groups.items()
                }
                for mn, groups in aggregated.items()
            }
            with open(agg_path, "w") as f:
                json.dump(agg_save, f, indent=2)

            # compare_groups_statistically는 all_results에 __stat_tests__를 삽입하지 않지만
            # 방어적으로 모델 키만 추출한 뷰를 넘겨 aggregated 원본을 보호
            model_keys = [k for k in aggregated if not k.startswith("__")]
            if len(model_keys) > 1:
                print(f"\n[{label}] 통합 통계 검정 ({len(model_keys)}개 모델)...")
                stat_view = {k: aggregated[k] for k in model_keys}
                compare_groups_statistically(
                    all_results = stat_view,
                    k_values    = DEFAULT_K_VALUES,
                    output_dir  = xai_subdir,
                )
            return aggregated

        # ── 11a-b. 홀드아웃 10%
        all_xai_holdout = _run_xai_multiset(
            val_dataset   = test_b6aepl,
            group_indices = group_indices_aepl,
            xai_subdir    = mkdir(os.path.join(xai_dir, "holdout")),
            label         = "Holdout 10%",
        )

        # ── 11c-d. CASF-2016
        all_xai_casf = {}
        if casf_aepl and group_indices_casf_aepl:
            all_xai_casf = _run_xai_multiset(
                val_dataset   = casf_aepl,
                group_indices = group_indices_casf_aepl,
                xai_subdir    = mkdir(os.path.join(xai_dir, "casf2016")),
                label         = "CASF-2016",
            )

        # ────────────────────────────────────────────────────────────
        # Step 12: 시각화
        # ────────────────────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print("Step 12. 시각화 생성")
        print(f"{'─'*50}")

        def _plot_xai_set(xai_dict, viz_subdir, label):
            if not xai_dict:
                return
            # __stat_tests__ 등 비모델 키가 섞이면 plot 함수가 잘못된 구조를 받아 크래시
            model_only = {k: v for k, v in xai_dict.items()
                          if not k.startswith("__")}
            for model_name, group_stats in model_only.items():
                mviz = mkdir(os.path.join(viz_subdir, model_name))
                plot_edge_barchart(
                    group_stats_dict=group_stats, model_name=f"{model_name} [{label}]",
                    k=25, save_path=os.path.join(mviz, "edge_barchart_top25.png"),
                )
                plot_edge_barchart(
                    group_stats_dict=group_stats, model_name=f"{model_name} [{label}]",
                    k=10, save_path=os.path.join(mviz, "edge_barchart_top10.png"),
                )
                plot_topk_lineplot(
                    group_stats_dict=group_stats, model_name=f"{model_name} [{label}]",
                    k_values=DEFAULT_K_VALUES,
                    save_path=os.path.join(mviz, "topk_lineplot.png"),
                )
            if len(model_only) > 1:
                plot_multi_model_comparison(
                    all_results=model_only, k=25,
                    save_path=os.path.join(viz_subdir, "multi_model_comparison_top25.png"),
                )

        _plot_xai_set(all_xai_holdout, mkdir(os.path.join(viz_dir, "holdout")),  "Holdout 10%")
        _plot_xai_set(all_xai_casf,   mkdir(os.path.join(viz_dir, "casf2016")), "CASF-2016")

    # ────────────────────────────────────────────────────────────────
    # 완료
    # ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  파이프라인 완료! 총 소요 시간: {elapsed/60:.1f}분")
    print(f"  결과 위치: {args.output_dir}")
    print(f"{'='*60}")

    with open(os.path.join(args.output_dir, "pipeline_meta.json"), "w") as f:
        json.dump({
            "elapsed_minutes": round(elapsed / 60, 2),
            "device":          str(device),
            "args":            vars(args),
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
