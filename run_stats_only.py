"""
저장된 Shapley CSV 파일로 통계 분석만 재실행하는 스크립트.
사용법: python run_stats_only.py --xai_dir <xai/holdout 경로> --output_dir <결과 저장 경로>
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "xai_analyzer",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline", "xai_analyzer.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compare_groups_statistically = _mod.compare_groups_statistically

TOPK_LIST = [5, 10, 15, 20, 25]
MODELS = [
    "GEMS_B6AEPL_CleanSplit",
    "GEMS_B6AEPL_PDBbind",
    "GC_GNN_CleanSplit",
    "GC_GNN_PDBbind",
]
GROUPS = ["low", "medium", "high"]


def compute_topk_stats(df: pd.DataFrame) -> dict:
    """Shapley CSV 한 샘플로 topk_stats 계산."""
    total = len(df)
    if total == 0:
        return {}

    baseline = (df["type"] == "interaction").sum() / total
    df_sorted = df.reindex(df["abs_shapley"].abs().sort_values(ascending=False).index)

    stats = {}
    for k in TOPK_LIST:
        top = df_sorted.head(k)
        n_inter = (top["type"] == "interaction").sum()
        n_lig   = (top["type"] == "ligand").sum()
        n_prot  = (top["type"] == "protein").sum()
        ratio_i = n_inter / k
        ratio_l = n_lig   / k
        ratio_p = n_prot  / k
        lift_i  = ratio_i / baseline if baseline > 0 else float("nan")
        lift_l  = ratio_l / ((df["type"] == "ligand").sum()   / total) if (df["type"] == "ligand").sum() > 0 else float("nan")
        lift_p  = ratio_p / ((df["type"] == "protein").sum()  / total) if (df["type"] == "protein").sum() > 0 else float("nan")
        stats[k] = {
            "interaction":          ratio_i,
            "ligand":               ratio_l,
            "protein":              ratio_p,
            "baseline_interaction": baseline,
            "lift_interaction":     lift_i,
            "lift_ligand":          lift_l,
            "lift_protein":         lift_p,
        }
    return stats


def load_seed_results(seed_dir: str) -> dict:
    """seed_dir 아래 모든 모델·그룹의 per_sample 데이터 로드."""
    all_results = {}
    for model in MODELS:
        model_dir = os.path.join(seed_dir, model)
        if not os.path.isdir(model_dir):
            print(f"[skip] {model} 디렉토리 없음")
            continue
        all_results[model] = {}
        for grp in GROUPS:
            grp_dir = os.path.join(model_dir, grp)
            if not os.path.isdir(grp_dir):
                print(f"[skip] {model}/{grp} 없음")
                continue
            per_sample = []
            for pdb_id in sorted(os.listdir(grp_dir)):
                sample_dir = os.path.join(grp_dir, pdb_id)
                if not os.path.isdir(sample_dir):
                    continue
                csv_path = os.path.join(sample_dir, f"{pdb_id}_shapley.csv")
                if not os.path.exists(csv_path):
                    continue
                df = pd.read_csv(csv_path)
                topk = compute_topk_stats(df)
                if topk:
                    per_sample.append({"id": pdb_id, "topk_stats": topk})
            all_results[model][grp] = {"per_sample": per_sample}
            print(f"  {model}/{grp}: {len(per_sample)} 샘플 로드")
    return all_results


def _json_safe(obj):
    if isinstance(obj, dict):  return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj: return None  # NaN
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")): return None
    try:
        import numpy as np
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
    except ImportError:
        pass
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xai_dir",    required=True, help="xai/holdout 경로")
    parser.add_argument("--output_dir", required=True, help="결과 저장 경로")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seed_dirs = sorted([
        d for d in os.listdir(args.xai_dir)
        if os.path.isdir(os.path.join(args.xai_dir, d)) and d.startswith("seed")
    ])
    print(f"발견된 seed 폴더: {seed_dirs}")

    for seed_name in seed_dirs:
        seed_dir = os.path.join(args.xai_dir, seed_name)
        print(f"\n=== {seed_name} 처리 중 ===")
        all_results = load_seed_results(seed_dir)

        out_dir = os.path.join(args.output_dir, seed_name)
        os.makedirs(out_dir, exist_ok=True)

        stat_results = compare_groups_statistically(all_results, output_dir=None)

        out_path = os.path.join(out_dir, "stat_tests.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(stat_results), f, indent=2, ensure_ascii=False)
        print(f"[완료] {out_path} 저장")


if __name__ == "__main__":
    main()
