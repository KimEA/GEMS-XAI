"""
Module 5: Visualizer
=====================
연구 결과의 시각화 모듈.

1. 성능 비교 막대 그래프:  GEMS/GCN × PDBbind/CleanSplit RMSE·R 비교
2. 엣지 타입 막대 그래프:  각 친화도 그룹별 Top-25 엣지 타입 비율
3. Top-k 선 그래프:        k=5~25에 따른 엣지 타입 비율 추이
4. RDKit 2D 리간드 시각화: Shapley 값 기반 결합 중요 엣지 색상 표현
5. PyMOL 스크립트 생성:    3D 단백질-리간드 복합체 위에 중요 상호작용 시각화

의존성:
    - matplotlib, numpy: 1~3
    - rdkit: 4 (conda: rdkit)
    - PyMOL: 5 (외부 설치 필요, 스크립트 생성만 담당)
"""

import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional


# ─── 색상 팔레트 ──────────────────────────────────────────────────────────────

EDGE_COLORS = {
    "ligand":      "#4C72B0",   # 파란 계열 (공유결합)
    "interaction": "#DD8452",   # 주황 계열 (비공유 상호작용)
    "protein":     "#55A868",   # 녹색 계열 (단백질 내부)
}
GROUP_COLORS = {
    "low":    "#74C476",   # 연두 (약한 결합)
    "medium": "#FD8D3C",   # 주황 (중간 결합)
    "high":   "#E6550D",   # 진한 주황-빨강 (강한 결합)
}
GROUP_LABELS = {
    "low":    "Low affinity\n(pKi < 6.0)",
    "medium": "Medium affinity\n(6.5 ≤ pKi ≤ 7.5)",
    "high":   "High affinity\n(pKi > 8.0)",
}
MODEL_MARKERS = {
    "GEMS_CleanSplit": ("o", "#1F77B4"),
    "GEMS_PDBbind":    ("s", "#AEC7E8"),
    "GCN_CleanSplit":  ("^", "#FF7F0E"),
    "GCN_PDBbind":     ("D", "#FFBB78"),
}


# ─── 1. 성능 비교 막대 그래프 ──────────────────────────────────────────────────

def plot_performance_comparison(
    metrics_dict: Dict[str, dict],
    save_path: str,
    title: str = "Model Performance on CASF-2016",
):
    """
    GEMS/GCN × PDBbind/CleanSplit 모델 성능 비교 막대 그래프.

    Args:
        metrics_dict: {모델명: {"RMSE": float, "R": float, "R²": float}}
        save_path:    저장 경로 (.png)
        title:        그래프 제목

    예시:
        metrics_dict = {
            "GEMS_CleanSplit": {"RMSE": 1.20, "R": 0.86, "R²": 0.74},
            "GEMS_PDBbind":    {"RMSE": 1.35, "R": 0.82, "R²": 0.67},
            "GCN_CleanSplit":  {"RMSE": 1.45, "R": 0.79, "R²": 0.62},
            "GCN_PDBbind":     {"RMSE": 1.58, "R": 0.75, "R²": 0.56},
        }
    """
    model_names = list(metrics_dict.keys())
    rmse_vals   = [metrics_dict[m]["RMSE"] for m in model_names]
    r_vals      = [metrics_dict[m]["R"]    for m in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(model_names))
    width = 0.5

    # 색상: CleanSplit = 진하게, PDBbind = 연하게
    colors = []
    for name in model_names:
        if "GEMS" in name and "CleanSplit" in name:
            colors.append("#1F77B4")
        elif "GEMS" in name and "PDBbind" in name:
            colors.append("#AEC7E8")
        elif "GCN" in name and "CleanSplit" in name:
            colors.append("#FF7F0E")
        elif "GCN" in name and "PDBbind" in name:
            colors.append("#FFBB78")
        else:
            colors.append("#999999")

    # RMSE 막대 그래프 (낮을수록 좋음)
    bars1 = ax1.bar(x, rmse_vals, width, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("RMSE (pK units)")
    ax1.set_title("RMSE ↓")
    ax1.set_ylim(0, max(rmse_vals) * 1.3)
    for bar, val in zip(bars1, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Pearson R 막대 그래프 (높을수록 좋음)
    bars2 = ax2.bar(x, r_vals, width, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Pearson R")
    ax2.set_title("Pearson R ↑")
    ax2.set_ylim(0, 1.0)
    for bar, val in zip(bars2, r_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # 범례
    legend_patches = [
        mpatches.Patch(color="#1F77B4", label="GEMS (CleanSplit)"),
        mpatches.Patch(color="#AEC7E8", label="GEMS (PDBbind)"),
        mpatches.Patch(color="#FF7F0E", label="GCN  (CleanSplit)"),
        mpatches.Patch(color="#FFBB78", label="GCN  (PDBbind)"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] 성능 비교 저장: {save_path}")


# ─── 2. 엣지 타입 막대 그래프 (Top-25) ────────────────────────────────────────

def plot_edge_barchart(
    group_stats_dict: Dict[str, dict],
    model_name:  str,
    k:           int = 25,
    save_path:   str = None,
):
    """
    각 친화도 그룹(Low/Medium/High)별 Top-k 엣지 중
    Ligand / Interaction / Protein 비율을 스택 막대 그래프로 시각화.

    Args:
        group_stats_dict: {그룹명: run_xai_for_group() 반환값}
        model_name:       그래프 제목에 사용할 모델 이름
        k:                분석 기준 k 값 (기본 25)
        save_path:        저장 경로 (.png)
    """
    groups     = list(group_stats_dict.keys())
    n_groups   = len(groups)

    lig_ratios  = []
    int_ratios  = []
    prot_ratios = []

    for grp in groups:
        stats = group_stats_dict[grp]
        d     = stats["mean_topk"].get(k, {})
        lig_ratios.append(d.get("ligand",      0.0) * 100)
        int_ratios.append(d.get("interaction", 0.0) * 100)
        prot_ratios.append(d.get("protein",    0.0) * 100)

    x     = np.arange(n_groups)
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))

    # 스택 막대 그래프
    bars_lig  = ax.bar(x, lig_ratios,  width, color=EDGE_COLORS["ligand"],
                        label="Ligand edge (공유결합)")
    bars_int  = ax.bar(x, int_ratios,  width, bottom=lig_ratios,
                        color=EDGE_COLORS["interaction"],
                        label="Interaction edge (비공유 상호작용)")
    bars_prot = ax.bar(x, prot_ratios, width,
                        bottom=[l + i for l, i in zip(lig_ratios, int_ratios)],
                        color=EDGE_COLORS["protein"],
                        label="Protein edge (단백질 내부)")

    # 값 레이블 표시 (Ligand, Interaction)
    for i, (l, it, p) in enumerate(zip(lig_ratios, int_ratios, prot_ratios)):
        if l > 2:
            ax.text(i, l / 2, f"{l:.1f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        if it > 2:
            ax.text(i, l + it / 2, f"{it:.1f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups], fontsize=10)
    ax.set_ylabel("엣지 타입 비율 (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title(
        f"[{model_name}] Top-{k} 엣지 타입 분포 by 결합 친화도 그룹",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Visualizer] 엣지 타입 막대 그래프 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ─── 3. Top-k 선 그래프 ───────────────────────────────────────────────────────

def plot_topk_lineplot(
    group_stats_dict: Dict[str, dict],
    model_name:  str,
    k_values:    List[int] = None,
    save_path:   str = None,
):
    """
    k값이 5 → 25로 증가함에 따른 각 그룹별 엣지 타입 비율 추이 선 그래프.

    각 행 = 엣지 타입 (Ligand, Interaction, Protein)
    각 선 = 친화도 그룹 (Low, Medium, High)

    Args:
        group_stats_dict: {그룹명: run_xai_for_group() 반환값}
        model_name:       그래프 제목에 사용
        k_values:         x축 k 값 목록 (기본 [5, 10, 15, 20, 25])
        save_path:        저장 경로
    """
    if k_values is None:
        k_values = [5, 10, 15, 20, 25]

    edge_type_labels = {
        "ligand":      "Ligand edge (공유결합)",
        "interaction": "Interaction edge (비공유 상호작용)",
        "protein":     "Protein edge (단백질 내부)",
    }
    edge_types = list(edge_type_labels.keys())
    groups     = list(group_stats_dict.keys())

    fig, axes = plt.subplots(1, len(edge_types), figsize=(15, 5), sharey=True)

    for ax, etype in zip(axes, edge_types):
        for grp in groups:
            stats = group_stats_dict[grp]
            ratios = []
            for k in k_values:
                d = stats["mean_topk"].get(k, {})
                ratios.append(d.get(etype, 0.0) * 100)

            color  = GROUP_COLORS.get(grp, "#999999")
            label  = GROUP_LABELS.get(grp, grp)
            ax.plot(k_values, ratios, marker="o", color=color, linewidth=2,
                    markersize=7, label=label)

            # 표준편차 음영 (있을 때만)
            std_key = f"std_{etype}"
            stds = []
            for k in k_values:
                d = stats["mean_topk"].get(k, {})
                stds.append(d.get(std_key, 0.0) * 100)
            if any(s > 0 for s in stds):
                ax.fill_between(
                    k_values,
                    [r - s for r, s in zip(ratios, stds)],
                    [r + s for r, s in zip(ratios, stds)],
                    alpha=0.15, color=color
                )

        ax.set_xlabel("Top-k (k 값)", fontsize=10)
        ax.set_ylabel("비율 (%)", fontsize=10)
        ax.set_title(edge_type_labels[etype], fontsize=11)
        ax.set_xticks(k_values)
        ax.set_ylim(-5, 105)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"[{model_name}] k값에 따른 Top-k 엣지 타입 비율 추이",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Visualizer] Top-k 선 그래프 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ─── 4. RDKit 2D 리간드 시각화 ────────────────────────────────────────────────

def visualize_ligand_rdkit(
    sdf_path:   str,
    phi_edges:  List[float],
    edge_index: "torch.Tensor",
    n_lig:      int,
    save_path:  str,
    top_k:      int = 25,
):
    """
    RDKit을 사용해 리간드 2D 구조 위에 엣지 Shapley 값을 색상으로 표현.

    중요도에 따른 색상:
        - 양의 Shapley (결합에 기여): 빨간색 계열
        - 음의 Shapley (결합에 방해): 파란색 계열
        - Top-k 이내 리간드 엣지만 강조 표시

    Args:
        sdf_path:   리간드 SDF 파일 경로
        phi_edges:  각 엣지의 Shapley 값 (n_edges)
        edge_index: [2, n_edges] 텐서
        n_lig:      리간드 원자 수 (node_type 분류 기준)
        save_path:  저장 경로 (.png)
        top_k:      강조할 상위 k개 엣지

    의존성: rdkit (conda install -c conda-forge rdkit)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, rdMolDescriptors
        from rdkit.Chem.Draw import rdMolDraw2D
        from IPython.display import Image
    except ImportError:
        print("[Visualizer] RDKit이 설치되어 있지 않습니다. (conda install -c conda-forge rdkit)")
        return

    mol = Chem.SDMolSupplier(sdf_path, removeHs=True)[0]
    if mol is None:
        print(f"[Visualizer] SDF 파일 로드 실패: {sdf_path}")
        return

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    # 리간드 내부 엣지만 필터링 (양방향 → 단방향으로 집계)
    bond_shap = {}  # (min_atom, max_atom) → Shapley 합
    for i, (s, d, phi) in enumerate(zip(src, dst, phi_edges)):
        if s < n_lig and d < n_lig and s != d:
            key = (min(int(s), int(d)), max(int(s), int(d)))
            bond_shap[key] = bond_shap.get(key, 0.0) + phi

    # Top-k 선택
    sorted_bonds = sorted(bond_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

    # 원자 및 결합 색상 딕셔너리 생성
    atom_colors  = {}
    bond_colors  = {}
    highlight_atoms = set()
    highlight_bonds = []

    for (a1, a2), shap_val in sorted_bonds:
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is None:
            continue
        bidx = bond.GetIdx()

        # 양/음 Shapley에 따른 색상 (RGB 튜플)
        intensity = min(abs(shap_val) / (max(abs(v) for _, v in sorted_bonds) + 1e-8), 1.0)
        if shap_val > 0:
            color = (1.0, 1.0 - intensity * 0.8, 1.0 - intensity * 0.8)  # 빨강 계열
        else:
            color = (1.0 - intensity * 0.8, 1.0 - intensity * 0.8, 1.0)  # 파랑 계열

        bond_colors[bidx]  = color
        atom_colors[a1]    = color
        atom_colors[a2]    = color
        highlight_atoms.add(a1)
        highlight_atoms.add(a2)
        highlight_bonds.append(bidx)

    # RDKit 2D 렌더링
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 400)
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(
        mol,
        highlightAtoms  = list(highlight_atoms),
        highlightBonds  = highlight_bonds,
        highlightAtomColors = atom_colors,
        highlightBondColors = bond_colors,
    )
    drawer.FinishDrawing()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(drawer.GetDrawingText())
    print(f"[Visualizer] RDKit 리간드 시각화 저장: {save_path}")


# ─── 5. PyMOL 스크립트 생성 ───────────────────────────────────────────────────

def generate_pymol_script(
    complex_id:  str,
    phi_edges:   List[float],
    edge_index:  "torch.Tensor",
    n_lig:       int,
    pdb_dir:     str,
    sdf_dir:     str,
    save_path:   str,
    top_k:       int = 25,
    pos:         Optional["torch.Tensor"] = None,
):
    """
    XAI 결과를 3D 구조 위에 시각화하는 PyMOL 스크립트(.pml) 생성.

    생성된 .pml 파일을 PyMOL에서 실행하면:
        - 단백질 구조 (회색 cartoon)
        - 리간드 구조 (cyan sticks)
        - 양의 Shapley Top-k 상호작용 엣지 → 빨간 점선
        - 음의 Shapley Top-k 상호작용 엣지 → 파란 점선
        - Top-k 리간드 엣지 → 노란 실선

    Args:
        complex_id:  PDB 코드 (예: '1abc')
        phi_edges:   각 엣지의 Shapley 값
        edge_index:  [2, n_edges] 텐서
        n_lig:       리간드 원자 수
        pdb_dir:     단백질 PDB 파일이 있는 디렉터리
        sdf_dir:     리간드 SDF 파일이 있는 디렉터리
        save_path:   생성할 .pml 파일 경로
        top_k:       시각화할 엣지 수
        pos:         원자 3D 좌표 [N, 3] (있으면 거리 기반 표시 가능)

    ⚠️ 참고:
        - PDB 및 SDF 파일은 PDBbind 데이터베이스에서 직접 다운로드 필요
        - pos가 없으면 연결 정보만 표시 (실제 3D 위치 없이)
        - PyMOL 설치 필요: https://pymol.org
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    # 엣지 분류 및 Top-k 선택
    interact_edges = []
    ligand_edges   = []
    for i, (s, d, phi) in enumerate(zip(src, dst, phi_edges)):
        if s == d:
            continue
        s_lig = s < n_lig
        d_lig = d < n_lig
        if s_lig != d_lig:                     # 상호작용 엣지
            interact_edges.append((i, s, d, phi))
        elif s_lig and d_lig:                   # 리간드 내부 엣지
            ligand_edges.append((i, s, d, phi))

    # 절댓값 기준 Top-k 선택
    interact_topk = sorted(interact_edges, key=lambda x: abs(x[3]), reverse=True)[:top_k]
    lig_topk      = sorted(ligand_edges,   key=lambda x: abs(x[3]), reverse=True)[:top_k]

    # 최대 Shapley 값 (정규화용)
    all_top = interact_topk + lig_topk
    max_abs  = max((abs(x[3]) for x in all_top), default=1.0)

    pdb_path = os.path.join(pdb_dir, f"{complex_id}_protein.pdb")
    sdf_path = os.path.join(sdf_dir, f"{complex_id}_ligand.sdf")

    lines = [
        "# ═══════════════════════════════════════════════════",
        f"# PyMOL 시각화 스크립트: {complex_id}",
        "# XAI (EdgeSHAPer) Top-k 중요 엣지 시각화",
        "# ═══════════════════════════════════════════════════",
        "",
        "# ─ 환경 설정",
        "reinitialize",
        "bg_color white",
        "set ray_shadows, 0",
        "",
        "# ─ 구조 파일 로드",
        f'load {pdb_path}, protein',
        f'load {sdf_path}, ligand',
        "",
        "# ─ 시각화 스타일",
        "hide everything",
        "show cartoon, protein",
        "show sticks, ligand",
        "color grey80, protein",
        "color cyan, ligand",
        "set cartoon_transparency, 0.3",
        "",
        "# ─ 리간드 주변 단백질 잔기 강조",
        "select binding_site, (protein within 5.0 of ligand)",
        "show sticks, binding_site",
        "color wheat, binding_site",
        "",
    ]

    # ─ 상호작용 엣지 (Interaction): 빨간/파란 점선
    lines.append("# ─ Top-k 상호작용 엣지 (비공유 상호작용)")
    for rank, (edge_idx, s, d, phi) in enumerate(interact_topk):
        norm_intensity = abs(phi) / max_abs
        color_name = f"pos_interact_{rank}" if phi > 0 else f"neg_interact_{rank}"
        r = norm_intensity
        b = 0.0 if phi > 0 else norm_intensity
        g = 0.0

        if pos is not None:
            # 원자 좌표가 있으면 거리 기반 실선 표시
            s_pos = pos[s].numpy()
            d_pos = pos[d].numpy()
            lines.append(
                f"distance {complex_id}_interact_{rank}, "
                f"id {s+1}, id {d+1}"
            )
            lines.append(f"color {('red' if phi > 0 else 'blue')}, {complex_id}_interact_{rank}")
        else:
            # 좌표 없이 의사코드로 표기
            lines.append(f"# Edge {rank+1}: atom_{s} ↔ atom_{d}  |φ|={abs(phi):.4f}  "
                        f"({'positive' if phi > 0 else 'negative'})")

    # ─ 리간드 내부 엣지: 노란 실선
    lines.append("\n# ─ Top-k 리간드 엣지 (공유결합 중요도)")
    for rank, (edge_idx, s, d, phi) in enumerate(lig_topk):
        if pos is not None:
            lines.append(
                f"distance {complex_id}_lig_{rank}, id {s+1}, id {d+1}"
            )
            lines.append(f"color yellow, {complex_id}_lig_{rank}")
        else:
            lines.append(f"# LigEdge {rank+1}: atom_{s} ↔ atom_{d}  |φ|={abs(phi):.4f}")

    lines += [
        "",
        "# ─ 거리 표시 스타일 설정 (실선)",
        "set dash_gap, 0, all",
        "set dash_width, 3.0, all",
        "",
        "# ─ 범례 (PyMOL 콘솔에서 수동 확인)",
        "# 빨간  점선: 양의 Shapley (결합 친화도 증가 기여)",
        "# 파란  점선: 음의 Shapley (결합 친화도 감소 기여)",
        "# 노란  실선: Top-k 리간드 결합",
        "",
        "# ─ 화면 설정 및 저장",
        "orient ligand",
        "zoom ligand, 5",
        f"png {os.path.splitext(save_path)[0]}_view.png, width=1200, height=900, dpi=300",
        "",
        "# ─ PyMOL 세션 저장",
        f"save {os.path.splitext(save_path)[0]}.pse",
    ]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Visualizer] PyMOL 스크립트 생성: {save_path}")
    print(f"  실행 방법: pymol {save_path}")
    print(f"  ⚠  PDB: {pdb_path}")
    print(f"  ⚠  SDF: {sdf_path}")


# ─── 여러 모델 비교용 멀티패널 그래프 ─────────────────────────────────────────

def plot_multi_model_comparison(
    all_results:  dict,
    k:            int = 25,
    save_path:    str = None,
):
    """
    여러 모델(GEMS/GCN × PDBbind/CleanSplit)의 동일 k에서
    그룹별 Interaction 엣지 비율을 비교하는 그룹 막대 그래프.

    Args:
        all_results: run_full_xai_analysis() 반환값
                     {모델명: {그룹명: group_stats}}
        k:           분석 기준 k 값
        save_path:   저장 경로
    """
    model_names  = list(all_results.keys())
    group_names  = ["low", "medium", "high"]
    n_models     = len(model_names)
    n_groups     = len(group_names)

    interact_ratios = np.zeros((n_models, n_groups))
    for mi, mname in enumerate(model_names):
        for gi, gname in enumerate(group_names):
            if gname in all_results[mname]:
                d = all_results[mname][gname]["mean_topk"].get(k, {})
                interact_ratios[mi, gi] = d.get("interaction", 0.0) * 100

    x     = np.arange(n_groups)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, mname in enumerate(model_names):
        offset  = (mi - n_models / 2 + 0.5) * width
        _, color = MODEL_MARKERS.get(mname, ("o", "#999999"))
        bars = ax.bar(x + offset, interact_ratios[mi], width, label=mname,
                      color=color, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, interact_ratios[mi]):
            if val > 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in group_names], fontsize=10)
    ax.set_ylabel(f"Interaction 엣지 비율 (%) in Top-{k}", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title(
        f"모델 비교: Interaction 엣지 비율 (Top-{k})\n"
        f"높을수록 물리화학적 상호작용에 집중",
        fontsize=12, fontweight="bold"
    )
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Visualizer] 멀티 모델 비교 그래프 저장: {save_path}")
    else:
        plt.show()
    plt.close()
