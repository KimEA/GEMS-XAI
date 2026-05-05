"""
holdout 샘플의 엣지 거리 정보를 Shapley CSV에 추가하는 스크립트.
서버에서 실행: python extract_edge_distances.py

edge_attr[:, 3] = 정규화된 거리 (× 10 → Å 단위 복원)
실제 결합 접촉 기준: ≤ 3.5Å (수소결합), ≤ 4.0Å (비공유결합 일반)
"""
import os, json, torch
import pandas as pd

RESULTS_DIR  = "/workspace/GEMS-XAI/results/pipeline"
XAI_DIR      = os.path.join(RESULTS_DIR, "xai/holdout")
SPLIT_JSON   = os.path.join(RESULTS_DIR, "id_split.json")
DATASET_PATH = "/workspace/GEMS_pytorch_datasets/B6AEPL_train_cleansplit.pt"
DIST_COL_IDX = 3   # edge_attr의 거리 컬럼 인덱스
DIST_SCALE   = 10  # 정규화 역변환 (× 10 → Å)


def load_dataset(path):
    print(f"[load] {path} ...")
    data = torch.load(path, map_location="cpu")
    # GEMS 데이터셋은 리스트 또는 InMemoryDataset
    if hasattr(data, "__iter__"):
        graphs = list(data)
    else:
        graphs = [data[i] for i in range(len(data))]
    print(f"  {len(graphs)} 샘플 로드 완료")
    return graphs


def build_id_map(graphs):
    return {g.id: g for g in graphs}


def add_distance_to_csv(csv_path, graph):
    df = pd.read_csv(csv_path)
    if "distance_A" in df.columns:
        return  # 이미 처리됨

    edge_attr = graph.edge_attr  # [N_edges, 20]
    if edge_attr is None or edge_attr.shape[1] <= DIST_COL_IDX:
        df["distance_A"] = float("nan")
    else:
        dist_raw = edge_attr[:, DIST_COL_IDX].numpy()
        dist_A   = dist_raw * DIST_SCALE
        # edge_idx 컬럼으로 매핑
        df["distance_A"] = df["edge_idx"].apply(
            lambda i: float(dist_A[i]) if i < len(dist_A) else float("nan")
        )

    df.to_csv(csv_path, index=False)


def main():
    # holdout ID 목록 로드
    with open(SPLIT_JSON) as f:
        split = json.load(f)
    holdout_ids = set(split.get("test", []))
    print(f"[split] holdout 샘플 수: {len(holdout_ids)}")

    # 데이터셋 로드 및 ID 맵 생성
    graphs = load_dataset(DATASET_PATH)
    id_map = build_id_map(graphs)

    # XAI CSV 순회하며 거리 컬럼 추가
    processed, skipped = 0, 0
    for seed_name in sorted(os.listdir(XAI_DIR)):
        seed_path = os.path.join(XAI_DIR, seed_name)
        if not os.path.isdir(seed_path):
            continue
        for model_name in sorted(os.listdir(seed_path)):
            model_path = os.path.join(seed_path, model_name)
            if not os.path.isdir(model_path):
                continue
            for grp in ["low", "medium", "high"]:
                grp_path = os.path.join(model_path, grp)
                if not os.path.isdir(grp_path):
                    continue
                for pdb_id in sorted(os.listdir(grp_path)):
                    csv_path = os.path.join(grp_path, pdb_id, f"{pdb_id}_shapley.csv")
                    if not os.path.exists(csv_path):
                        continue
                    if pdb_id not in id_map:
                        skipped += 1
                        continue
                    add_distance_to_csv(csv_path, id_map[pdb_id])
                    processed += 1

    print(f"\n완료: {processed}개 CSV 업데이트, {skipped}개 스킵(ID 없음)")


if __name__ == "__main__":
    main()
