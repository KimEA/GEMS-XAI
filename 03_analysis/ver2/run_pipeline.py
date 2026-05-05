"""
run_pipeline.py  ─  연구 파이프라인 메인 오케스트레이터
==========================================================
XAI 기반 GEMS 및 GCN 모델의 단백질-리간드 상호작용 학습 메커니즘 검증

실행 단계:
    Step 1. 데이터 로드 및 80:20 분할
    Step 2. SimpleGCN 학습 (PDBbind 버전 + CleanSplit 버전)
    Step 3. GEMS 앙상블 로드 (PDBbind 버전 + CleanSplit 버전)
    Step 4. CASF-2016 성능 평가 (RMSE, Pearson R)
    Step 5. 검증 셋 층화 (Low / Medium / High affinity)
    Step 6. EdgeSHAPer XAI 분석 (그룹 × 모델 × k값)
    Step 7. 결과 시각화 (막대, 선, 3D 스크립트)

사용 예시:
    # 전체 파이프라인 실행
    KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/pli_m1/bin/python run_pipeline.py \\
        --train_cleansplit ../../02_data/GEMS_pytorch_datasets/B6AEPL_train_cleansplit.pt \\
        --train_pdbbind    ../../02_data/GEMS_pytorch_datasets/B6AEPL_train_pdbbind.pt \\
        --test_casf2016    ../../02_data/GEMS_pytorch_datasets/B6AEPL_casf2016.pt \\
        --gems_model_dir   GEMS/model \\
        --output_dir       results/pipeline \\
        --M 30 --max_per_group 5

    # 학습 건너뛰고 XAI만 실행 (이미 학습된 GCN 모델 있을 때)
    KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/pli_m1/bin/python run_pipeline.py \\
        --skip_train \\
        --gcn_cleansplit_ckpt results/pipeline/checkpoints/gcn_cleansplit_best.pt \\
        --gcn_pdbbind_ckpt    results/pipeline/checkpoints/gcn_pdbbind_best.pt \\
        ...

경로 설정 (기본값):
    - 데이터: 03_analysis/ 에서 실행 시 ../../02_data/GEMS_pytorch_datasets/
    - 모델:   GEMS/model/
    - 결과:   results/pipeline/
"""

import os
import sys
import json
import time
import argparse
import torch

# 현재 파일 기준 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "GEMS"))

DATA_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "02_data", "GEMS_pytorch_datasets")

# ─── 파이프라인 모듈 임포트 ───────────────────────────────────────────────────
from pipeline.data_loader import (
    load_gems_dataset, split_train_val, get_dataset_info,
    stratify_by_affinity, create_dataloader, get_graphs_from_indices,
)
from pipeline.trainer import (
    SimpleGCN, train_model, load_gcn_checkpoint,
)
from pipeline.evaluator import (
    ModelWrapper, load_gems_ensemble, auto_load_gems_ensemble,
    evaluate_model, compare_all_models,
)
from pipeline.xai_analyzer import (
    run_full_xai_analysis, DEFAULT_K_VALUES,
)
from pipeline.visualizer import (
    plot_performance_comparison, plot_edge_barchart,
    plot_topk_lineplot, plot_multi_model_comparison,
    generate_pymol_script,
)


# ─── 인자 파싱 ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GEMS/GCN XAI 비교 파이프라인",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── 데이터 경로
    p.add_argument("--train_cleansplit",
        default=os.path.join(DATA_ROOT, "B6AEPL_train_cleansplit.pt"),
        help="CleanSplit 학습 데이터 (.pt)")
    p.add_argument("--train_pdbbind",
        default=os.path.join(DATA_ROOT, "B6AEPL_train_pdbbind.pt"),
        help="PDBbind 학습 데이터 (.pt)")
    p.add_argument("--test_casf2016",
        default=os.path.join(DATA_ROOT, "B6AEPL_casf2016.pt"),
        help="CASF-2016 테스트 데이터 (.pt)")

    # ── 모델 경로
    p.add_argument("--gems_model_dir",
        default=os.path.join(SCRIPT_DIR, "GEMS", "model"),
        help="GEMS 사전 학습 모델 stdict 디렉터리")
    p.add_argument("--gems_pdbbind_model_dir",
        default=None,
        help="PDBbind로 학습된 GEMS 모델 디렉터리 (없으면 동일 dir 사용)")

    # ── 결과 저장
    p.add_argument("--output_dir",
        default=os.path.join(SCRIPT_DIR, "results", "pipeline"),
        help="모든 결과 저장 루트 디렉터리")

    # ── 학습 제어
    p.add_argument("--skip_train", action="store_true",
        help="GCN 학습 단계 건너뜀 (기존 체크포인트 사용)")
    p.add_argument("--gcn_cleansplit_ckpt", default=None,
        help="CleanSplit GCN 체크포인트 (--skip_train 시 필요)")
    p.add_argument("--gcn_pdbbind_ckpt",    default=None,
        help="PDBbind GCN 체크포인트 (--skip_train 시 필요)")
    p.add_argument("--skip_eval",  action="store_true",
        help="평가 단계 건너뜀")
    p.add_argument("--skip_xai",   action="store_true",
        help="XAI 분석 단계 건너뜀")

    # ── 학습 하이퍼파라미터
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--patience",     type=int,   default=30)
    p.add_argument("--hidden",       type=int,   default=256,
        help="SimpleGCN 은닉 레이어 차원")
    p.add_argument("--n_layers",     type=int,   default=2,
        help="SimpleGCN GATv2Conv 레이어 수")

    # ── XAI 설정
    p.add_argument("--M",            type=int,   default=50,
        help="EdgeSHAPer Monte Carlo 샘플링 횟수 (많을수록 정확, 느림)")
    p.add_argument("--max_per_group",type=int,   default=None,
        help="그룹당 최대 XAI 분석 샘플 수 (None = 전체)")
    p.add_argument("--train_ratio",  type=float, default=0.8,
        help="학습/검증 분할 비율 (0.8 = 80% 학습, 20% 검증)")
    p.add_argument("--seed",         type=int,   default=42)

    # ── XAI 대상 모델 선택
    p.add_argument("--xai_models", nargs="+",
        default=["GEMS_CleanSplit", "GCN_CleanSplit"],
        choices=["GEMS_CleanSplit", "GEMS_PDBbind", "GCN_CleanSplit", "GCN_PDBbind"],
        help="XAI 분석 대상 모델 목록")

    return p.parse_args()


# ─── 헬퍼: 디렉터리 생성 ─────────────────────────────────────────────────────

def mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ─── Step 2: GCN 학습 ────────────────────────────────────────────────────────

def train_gcn_model(
    train_data,
    val_data,
    dataset_info: dict,
    config:       dict,
    save_dir:     str,
    model_name:   str,
    device,
) -> SimpleGCN:
    """
    SimpleGCN 인스턴스화 → 학습 → 체크포인트 저장.
    """
    model = SimpleGCN(
        in_channels = dataset_info["node_feat_dim"],
        edge_dim    = dataset_info["edge_feat_dim"],
        hidden      = config.get("hidden", 256),
        n_layers    = config.get("n_layers", 2),
        lig_emb_dim = dataset_info["lig_emb_dim"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Pipeline] {model_name} 파라미터 수: {n_params:,}")

    train_loader = create_dataloader(train_data, config["batch_size"], shuffle=True)
    val_loader   = create_dataloader(val_data,   config["batch_size"], shuffle=False)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=save_dir,
        model_name=model_name,
    )
    return model


# ─── 메인 파이프라인 ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t_start = time.time()

    # ── 디바이스 설정
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  GEMS/GCN XAI 비교 파이프라인 시작")
    print(f"  디바이스: {device}")
    print(f"  출력 디렉터리: {args.output_dir}")
    print(f"{'='*60}\n")

    ckpt_dir   = mkdir(os.path.join(args.output_dir, "checkpoints"))
    eval_dir   = mkdir(os.path.join(args.output_dir, "evaluation"))
    xai_dir    = mkdir(os.path.join(args.output_dir, "xai"))
    viz_dir    = mkdir(os.path.join(args.output_dir, "figures"))

    # ────────────────────────────────────────────────────────────────
    # Step 1: 데이터 로드 및 분할
    # ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Step 1. 데이터 로드 및 분할")
    print(f"{'─'*50}")

    # CleanSplit 데이터: 80% 학습 + 20% 검증(XAI용)
    cleansplit_ds = load_gems_dataset(args.train_cleansplit)
    info          = get_dataset_info(cleansplit_ds)
    train_clean, val_clean = split_train_val(cleansplit_ds, args.train_ratio, args.seed)

    # PDBbind 데이터: 전체를 GCN_PDBbind 학습에 사용
    pdbbind_ds = load_gems_dataset(args.train_pdbbind)

    # CASF-2016 테스트 셋
    casf2016_ds = load_gems_dataset(args.test_casf2016)

    # 검증 셋 층화 (XAI 분석용)
    print("\n[Step 1] 검증 셋(20%) 층화 샘플링...")
    group_indices = stratify_by_affinity(val_clean)

    # 층화 결과 저장
    strat_path = os.path.join(args.output_dir, "stratification_summary.json")
    with open(strat_path, "w") as f:
        json.dump({g: len(idx) for g, idx in group_indices.items()}, f, indent=2)

    # ────────────────────────────────────────────────────────────────
    # Step 2: SimpleGCN 학습
    # ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Step 2. SimpleGCN 학습")
    print(f"{'─'*50}")

    train_config = {
        "lr":          args.lr,
        "weight_decay": 1e-5,
        "epochs":      args.epochs,
        "patience":    args.patience,
        "batch_size":  args.batch_size,
        "scheduler":   True,
        "hidden":      args.hidden,
        "n_layers":    args.n_layers,
    }

    gcn_cleansplit = None
    gcn_pdbbind    = None

    if not args.skip_train:
        # ─ GCN CleanSplit 버전: 80% CleanSplit으로 학습
        print("\n[Step 2a] GCN_CleanSplit 학습 (80% CleanSplit)...")
        gcn_cleansplit = train_gcn_model(
            train_data=train_clean,
            val_data=val_clean,
            dataset_info=info,
            config=train_config,
            save_dir=ckpt_dir,
            model_name="gcn_cleansplit",
            device=device,
        )

        # ─ GCN PDBbind 버전: 전체 PDBbind로 학습
        print("\n[Step 2b] GCN_PDBbind 학습 (전체 PDBbind)...")
        # PDBbind의 임시 val_set: 마지막 5% 사용 (early stopping 기준)
        _, val_pdb = split_train_val(pdbbind_ds, train_ratio=0.95, seed=args.seed)
        gcn_pdbbind = train_gcn_model(
            train_data=pdbbind_ds,
            val_data=val_pdb,
            dataset_info=info,
            config=train_config,
            save_dir=ckpt_dir,
            model_name="gcn_pdbbind",
            device=device,
        )
    else:
        # 체크포인트에서 로드
        if args.gcn_cleansplit_ckpt:
            print(f"[Step 2] GCN_CleanSplit 로드: {args.gcn_cleansplit_ckpt}")
            gcn_cleansplit = load_gcn_checkpoint(args.gcn_cleansplit_ckpt, device)
        if args.gcn_pdbbind_ckpt:
            print(f"[Step 2] GCN_PDBbind 로드: {args.gcn_pdbbind_ckpt}")
            gcn_pdbbind = load_gcn_checkpoint(args.gcn_pdbbind_ckpt, device)

    # ────────────────────────────────────────────────────────────────
    # Step 3: GEMS 앙상블 로드
    # ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Step 3. GEMS 앙상블 로드")
    print(f"{'─'*50}")

    gems_cleansplit_models = auto_load_gems_ensemble(args.gems_model_dir, casf2016_ds, device)

    gems_pdbbind_models = None
    if args.gems_pdbbind_model_dir:
        print(f"\n[Step 3b] GEMS_PDBbind 로드: {args.gems_pdbbind_model_dir}")
        gems_pdbbind_models = auto_load_gems_ensemble(args.gems_pdbbind_model_dir, casf2016_ds, device)

    # ModelWrapper 딕셔너리 구성
    wrappers = {}
    wrappers["GEMS_CleanSplit"] = ModelWrapper(
        "GEMS_CleanSplit", gems_cleansplit_models, device, is_ensemble=True
    )
    if gems_pdbbind_models:
        wrappers["GEMS_PDBbind"] = ModelWrapper(
            "GEMS_PDBbind", gems_pdbbind_models, device, is_ensemble=True
        )
    if gcn_cleansplit:
        gcn_cleansplit.eval()
        wrappers["GCN_CleanSplit"] = ModelWrapper(
            "GCN_CleanSplit", [gcn_cleansplit], device, is_ensemble=False
        )
    if gcn_pdbbind:
        gcn_pdbbind.eval()
        wrappers["GCN_PDBbind"] = ModelWrapper(
            "GCN_PDBbind", [gcn_pdbbind], device, is_ensemble=False
        )

    print(f"\n[Step 3] 등록된 모델: {list(wrappers.keys())}")

    # ────────────────────────────────────────────────────────────────
    # Step 4: CASF-2016 성능 평가
    # ────────────────────────────────────────────────────────────────
    if not args.skip_eval:
        print(f"\n{'─'*50}")
        print("Step 4. CASF-2016 성능 평가")
        print(f"{'─'*50}")

        comparison_df = compare_all_models(wrappers, casf2016_ds, eval_dir, args.batch_size)

        # 성능 비교 막대 그래프
        metrics_dict = {
            name: {
                "RMSE": row["RMSE"],
                "R":    row["R"],
                "R²":   row["R²"],
            }
            for name, row in zip(comparison_df["모델"], comparison_df.to_dict("records"))
        }
        plot_performance_comparison(
            metrics_dict,
            save_path=os.path.join(viz_dir, "performance_comparison.png"),
        )

    # ────────────────────────────────────────────────────────────────
    # Step 5 + 6: XAI 분석 (EdgeSHAPer)
    # ────────────────────────────────────────────────────────────────
    if not args.skip_xai:
        print(f"\n{'─'*50}")
        print("Step 5-6. XAI 분석 (EdgeSHAPer)")
        print(f"{'─'*50}")

        # XAI 대상 모델 필터링
        xai_wrappers = {k: v for k, v in wrappers.items() if k in args.xai_models}
        if not xai_wrappers:
            print("[경고] XAI 대상 모델이 없습니다. --xai_models 확인")
        else:
            print(f"\n[Step 6] XAI 분석 대상: {list(xai_wrappers.keys())}")
            print(f"         M={args.M}, k_values={DEFAULT_K_VALUES}")
            print(f"         그룹당 최대 샘플: {args.max_per_group}")

            # 예상 소요 시간 안내
            avg_edges = 400
            sec_per_edge = 1.5   # M=10 기준 약 1.5초/엣지
            est_sec_per_graph = avg_edges * (args.M / 10) * sec_per_edge
            total_graphs = sum(
                min(len(idx), args.max_per_group or len(idx))
                for idx in group_indices.values()
            ) * len(xai_wrappers)
            print(f"         예상 소요 시간: 그래프당 ~{est_sec_per_graph/60:.0f}분, "
                  f"총 ~{total_graphs * est_sec_per_graph / 3600:.1f}시간")

            all_xai_results = run_full_xai_analysis(
                wrappers      = xai_wrappers,
                val_dataset   = val_clean,
                group_indices = group_indices,
                M             = args.M,
                k_values      = DEFAULT_K_VALUES,
                output_dir    = xai_dir,
                max_per_group = args.max_per_group,
                device        = device,
            )

            # ─────────────────────────────────────────────────────
            # Step 7: 시각화
            # ─────────────────────────────────────────────────────
            print(f"\n{'─'*50}")
            print("Step 7. 시각화 생성")
            print(f"{'─'*50}")

            for model_name, group_stats in all_xai_results.items():
                model_viz_dir = mkdir(os.path.join(viz_dir, model_name))

                # ─ 2. 막대 그래프 (Top-25)
                plot_edge_barchart(
                    group_stats_dict = group_stats,
                    model_name       = model_name,
                    k                = 25,
                    save_path        = os.path.join(model_viz_dir, "edge_barchart_top25.png"),
                )

                # ─ 3. 선 그래프 (k=5~25)
                plot_topk_lineplot(
                    group_stats_dict = group_stats,
                    model_name       = model_name,
                    k_values         = DEFAULT_K_VALUES,
                    save_path        = os.path.join(model_viz_dir, "topk_lineplot.png"),
                )

                # ─ 추가: Top-10 막대 그래프 (논문용 보조)
                plot_edge_barchart(
                    group_stats_dict = group_stats,
                    model_name       = model_name,
                    k                = 10,
                    save_path        = os.path.join(model_viz_dir, "edge_barchart_top10.png"),
                )

            # ─ 전체 모델 비교 (Interaction 엣지 비율)
            if len(all_xai_results) > 1:
                plot_multi_model_comparison(
                    all_results = all_xai_results,
                    k           = 25,
                    save_path   = os.path.join(viz_dir, "multi_model_interaction_top25.png"),
                )

    # ────────────────────────────────────────────────────────────────
    # 완료
    # ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  파이프라인 완료! 총 소요 시간: {elapsed/60:.1f}분")
    print(f"  결과 위치: {args.output_dir}")
    print(f"{'='*60}")

    # 실행 메타정보 저장
    meta = {
        "elapsed_minutes": round(elapsed / 60, 2),
        "device":          str(device),
        "args":            vars(args),
        "models_used":     list(wrappers.keys()),
    }
    with open(os.path.join(args.output_dir, "pipeline_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ─── PyMOL 스크립트 단독 생성 유틸리티 ───────────────────────────────────────

def generate_3d_scripts(
    xai_dir:    str,
    pdb_dir:    str,
    sdf_dir:    str,
    model_name: str = "GEMS_CleanSplit",
    group_name: str = "high",
    top_k:      int = 25,
):
    """
    저장된 XAI 결과에서 PyMOL 스크립트 일괄 생성.
    파이프라인 완료 후 별도로 호출 가능.

    Args:
        xai_dir:    run_pipeline.py가 저장한 xai/ 디렉터리
        pdb_dir:    단백질 PDB 파일 디렉터리 (PDBbind에서 다운로드)
        sdf_dir:    리간드 SDF 파일 디렉터리
        model_name: 대상 모델 이름
        group_name: 대상 친화도 그룹
        top_k:      시각화할 엣지 수
    """
    import glob
    import torch

    sample_dirs = glob.glob(os.path.join(xai_dir, model_name, group_name, "*"))
    print(f"[3D] {model_name}/{group_name} 그룹에서 {len(sample_dirs)}개 샘플 처리")

    for sample_dir in sample_dirs:
        complex_id = os.path.basename(sample_dir)
        csv_path   = os.path.join(sample_dir, f"{complex_id}_shapley.csv")
        if not os.path.exists(csv_path):
            continue

        # Shapley CSV 읽기
        import csv as csv_mod
        phi_edges  = []
        src_list   = []
        dst_list   = []
        n_lig      = None

        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                src_list.append(int(row["src"]))
                dst_list.append(int(row["dst"]))
                phi_edges.append(float(row["shapley"]))

        if not phi_edges:
            continue

        # edge_index 텐서 복원
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # n_lig 추정: src/dst에서 리간드 원자 인덱스 최대값 추정
        # (정확한 n_lig는 원본 그래프에서만 얻을 수 있음)
        # 여기서는 파일명 옆에 저장된 stats.txt에서 읽거나 추정
        stats_path = os.path.join(sample_dir, f"{complex_id}_xai_stats.txt")

        # PyMOL 스크립트 생성
        pymol_path = os.path.join(sample_dir, f"{complex_id}_pymol.pml")
        generate_pymol_script(
            complex_id = complex_id,
            phi_edges  = phi_edges,
            edge_index = edge_index,
            n_lig      = 20,           # 추정값 (실제 그래프에서 확인 필요)
            pdb_dir    = pdb_dir,
            sdf_dir    = sdf_dir,
            save_path  = pymol_path,
            top_k      = top_k,
        )

    print(f"[3D] PyMOL 스크립트 생성 완료: {os.path.join(xai_dir, model_name, group_name)}")


# ─── 진입점 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
