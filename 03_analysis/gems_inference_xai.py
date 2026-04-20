"""
GEMS Inference + XAI Analysis (EdgeSHAPer adaptation)

Research goal:
    Verify whether GEMS predicts binding affinity by learning true physicochemical
    interactions (protein-ligand cross edges) or merely exploiting statistical
    biases in the training data.

Key XAI analysis:
    - EdgeSHAPer adapted for GEMS ensemble (handles edge_attr alongside edge_index)
    - Classify edges: ligand-internal vs. protein-ligand cross
    - If cross edges dominate top-k important edges → physicochemical understanding
    - If ligand-internal edges dominate → potential statistical bias

Usage:
    python gems_inference_xai.py \
        --dataset_path ../../02_data/GEMS_pytorch_datasets/00AEPL_casf2016_indep.pt \
        --model_path ../GEMS/model \
        --n_samples 30 \
        --M 50 \
        --top_k 10 \
        --output_dir results/gems_xai
"""

import os
import sys
import glob
import csv
import json
import argparse
import numpy as np
from numpy.random import default_rng
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

# Add GEMS source to path
GEMS_DIR = os.path.join(os.path.dirname(__file__), "GEMS")
sys.path.insert(0, GEMS_DIR)

from model.GEMS18 import GEMS18d, GEMS18e


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, output, targets):
        return torch.sqrt(self.mse(output, targets))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ensemble(model_path, model_arch, dataset_id, node_feat_dim, edge_feat_dim, device):
    """Load 5-fold ensemble from state dict files."""
    patterns = [
        os.path.join(model_path, f"{model_arch}_{dataset_id}_*_f{f}_best_stdict.pt")
        for f in range(5)
    ]
    stdict_paths = []
    for pat in patterns:
        stdict_paths.extend(glob.glob(pat))
    stdict_paths = sorted(stdict_paths)

    if not stdict_paths:
        raise FileNotFoundError(
            f"No model state dicts found for {model_arch}/{dataset_id} in {model_path}"
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

    print(f"Loaded {len(models)} models ({model_arch}/{dataset_id})")
    return models


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

PK_MIN, PK_MAX = 0.0, 16.0


def unscale(t: torch.Tensor) -> torch.Tensor:
    return t * (PK_MAX - PK_MIN) + PK_MIN


@torch.no_grad()
def ensemble_predict(models, graphbatch, device):
    """Ensemble prediction for a DataLoader batch. Returns (y_true, y_pred, ids)."""
    graphbatch = graphbatch.to(device)
    preds = [m(graphbatch).view(-1) for m in models]
    pred = torch.mean(torch.stack(preds), dim=0)
    return graphbatch.y.cpu(), pred.cpu(), list(graphbatch.id)


def run_inference(models, loader, device):
    """Run full inference, return unscaled (y_true, y_pred, ids) and metrics."""
    y_true_all, y_pred_all, ids_all = [], [], []
    criterion = RMSELoss()

    for batch in loader:
        y_true, y_pred, ids = ensemble_predict(models, batch, device)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        ids_all.extend(ids)

    y_true = torch.cat(y_true_all)
    y_pred = torch.cat(y_pred_all)

    has_labels = bool((y_true > 0).any())

    if has_labels:
        y_true_pk = unscale(y_true)
        y_pred_pk = unscale(y_pred)
    else:
        y_true_pk = y_true
        y_pred_pk = unscale(y_pred)

    metrics = {}
    if has_labels:
        rmse = criterion(y_pred_pk, y_true_pk).item()
        r = float(np.corrcoef(y_true_pk.numpy(), y_pred_pk.numpy())[0, 1])
        ss_res = ((y_true_pk - y_pred_pk) ** 2).sum().item()
        ss_tot = ((y_true_pk - y_true_pk.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot
        metrics = {"RMSE": rmse, "R": r, "R2": r2}
        print(f"Inference → R={r:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    return y_true_pk, y_pred_pk, ids_all, metrics, has_labels


# ---------------------------------------------------------------------------
# Single-graph ensemble prediction (used inside EdgeSHAPer)
# ---------------------------------------------------------------------------

def _make_graphbatch(x, edge_index, edge_attr, lig_emb, device):
    """
    Build a minimal fake graphbatch that GEMS forward() accepts for a single graph.
    """
    from torch_geometric.data import Data, Batch

    data = Data(
        x=x.float(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.float(),
    )
    batch_obj = Batch.from_data_list([data])

    # lig_emb shape from dataset: [1, 384] (already batched for 1 graph)
    # GEMS18d forward: u=graphbatch.lig_emb must be [num_graphs, 384]
    if lig_emb is not None:
        emb = lig_emb.float()
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)   # [384] → [1, 384]
        batch_obj.lig_emb = emb.to(device)

    return batch_obj.to(device)


@torch.no_grad()
def single_graph_predict(models, x, edge_index, edge_attr, lig_emb, device):
    """
    Run ensemble on a single graph with (possibly perturbed) edge_index / edge_attr.
    Returns the mean predicted pK (scaled).
    """
    if edge_index.shape[1] == 0:
        # Degenerate: no edges → return 0 contribution
        return 0.0

    gb = _make_graphbatch(x, edge_index, edge_attr, lig_emb, device)
    preds = []
    for m in models:
        try:
            out = m(gb).view(-1)
            preds.append(out)
        except Exception:
            return 0.0  # skip if degenerate graph breaks model
    if not preds:
        return 0.0
    return torch.mean(torch.stack(preds)).item()


# ---------------------------------------------------------------------------
# GEMS-adapted EdgeSHAPer
# ---------------------------------------------------------------------------

def gems_edgeshaper(models, graph, M=50, seed=42, device="cpu"):
    """
    EdgeSHAPer for GEMS ensemble models.

    Unlike the original EdgeSHAPer (which uses model(x, edge_index, batch, edge_weight)),
    GEMS forward() also requires edge_attr and lig_emb. When edges are masked,
    edge_attr rows must be masked identically.

    Args:
        models:  list of GEMS model instances (ensemble)
        graph:   single PyG Data object from the GEMS dataset
        M:       number of Monte Carlo samples per edge
        seed:    RNG seed
        device:  torch device

    Returns:
        phi_edges: list of float Shapley values, one per edge (len == edge_index.shape[1])
    """
    x         = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    edge_attr  = graph.edge_attr.to(device)
    lig_emb    = graph.lig_emb.to(device) if (hasattr(graph, "lig_emb") and graph.lig_emb is not None) else None

    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]

    if num_edges == 0:
        return []

    max_possible = num_nodes * (num_nodes - 1)
    P = num_edges / max_possible if max_possible > 0 else 0.5

    rng = default_rng(seed)
    phi_edges = []

    for j in tqdm(range(num_edges), desc="EdgeSHAPer", leave=False):
        marginal = 0.0
        for _ in range(M):
            # Random binary mask for "background" graph z
            E_z_mask = rng.binomial(1, P, num_edges).astype(bool)
            # Random permutation of edges
            pi = torch.randperm(num_edges).numpy()
            j_pos = int(np.where(pi == j)[0][0])

            # S ∪ {j} mask: edges up to and including j in pi → present; rest → from z
            mask_plus = np.zeros(num_edges, dtype=bool)
            for k in range(num_edges):
                mask_plus[pi[k]] = True if k <= j_pos else E_z_mask[pi[k]]

            # S mask: edges strictly before j in pi → present; rest → from z
            mask_minus = np.zeros(num_edges, dtype=bool)
            for k in range(num_edges):
                mask_minus[pi[k]] = True if k < j_pos else E_z_mask[pi[k]]

            idx_plus  = torch.from_numpy(np.where(mask_plus)[0]).long().to(device)
            idx_minus = torch.from_numpy(np.where(mask_minus)[0]).long().to(device)

            ei_plus  = edge_index[:, idx_plus]
            ea_plus  = edge_attr[idx_plus]
            ei_minus = edge_index[:, idx_minus]
            ea_minus = edge_attr[idx_minus]

            v_plus  = single_graph_predict(models, x, ei_plus,  ea_plus,  lig_emb, device)
            v_minus = single_graph_predict(models, x, ei_minus, ea_minus, lig_emb, device)

            marginal += v_plus - v_minus

        phi_edges.append(marginal / M)

    return phi_edges


# ---------------------------------------------------------------------------
# Edge classification
# ---------------------------------------------------------------------------

def classify_edges(edge_index, n_lig_nodes):
    """
    Classify each edge as 'lig_internal' or 'cross' (protein-ligand).

    In GEMS graphs:
      - nodes 0 .. n_lig-1  are ligand atoms
      - nodes n_lig .. N-1  are protein residues
      - There are no protein-protein internal edges

    Returns:
        edge_types: list of str, one per edge
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    types = []
    for u, v in zip(src, dst):
        u_is_lig = u < n_lig_nodes
        v_is_lig = v < n_lig_nodes
        if u_is_lig and v_is_lig:
            types.append("lig_internal")
        else:
            types.append("cross")
    return types


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_scatter(y_true, y_pred, r, rmse, r2, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=18, c="steelblue")
    lim = 16
    ax.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("True pK")
    ax.set_ylabel("Predicted pK")
    ax.set_title(f"GEMS Inference\nR={r:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_edge_type_fractions(stats_list, top_k, save_path):
    """
    Bar chart: fraction of top-k important edges that are cross vs lig_internal,
    across all explained samples.
    """
    cross_fracs  = [s["cross_frac_topk"]  for s in stats_list]
    lig_fracs    = [s["lig_frac_topk"]    for s in stats_list]
    ids          = [s["id"]               for s in stats_list]

    x_pos = np.arange(len(ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.6), 5))
    ax.bar(x_pos - width/2, cross_fracs, width, label="Cross (protein-ligand)", color="coral")
    ax.bar(x_pos + width/2, lig_fracs,   width, label="Ligand-internal",        color="steelblue")
    ax.axhline(np.mean(cross_fracs), color="red",       linestyle="--", linewidth=1, label=f"Mean cross={np.mean(cross_fracs):.2f}")
    ax.axhline(np.mean(lig_fracs),   color="steelblue", linestyle="--", linewidth=1, label=f"Mean lig={np.mean(lig_fracs):.2f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"Fraction in top-{top_k} edges")
    ax.set_ylim(0, 1)
    ax.set_title(f"Edge-type distribution in top-{top_k} important edges (EdgeSHAPer)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_shapley_histogram(phi_edges, edge_types, sample_id, save_path):
    """Distribution of Shapley values by edge type for one sample."""
    cross_vals = [v for v, t in zip(phi_edges, edge_types) if t == "cross"]
    lig_vals   = [v for v, t in zip(phi_edges, edge_types) if t == "lig_internal"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 30
    if cross_vals:
        ax.hist(cross_vals, bins=bins, alpha=0.6, color="coral",     label="Cross (protein-ligand)")
    if lig_vals:
        ax.hist(lig_vals,   bins=bins, alpha=0.6, color="steelblue", label="Ligand-internal")
    ax.set_xlabel("Shapley value (edge importance)")
    ax.set_ylabel("Count")
    ax.set_title(f"{sample_id} — EdgeSHAPer value distribution by edge type")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_mean_shapley_by_type(stats_list, save_path):
    """Mean absolute Shapley value per edge type across all samples."""
    mean_cross = [s["mean_abs_cross"] for s in stats_list if s["n_cross"] > 0]
    mean_lig   = [s["mean_abs_lig"]   for s in stats_list if s["n_lig"]   > 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    data_to_plot = []
    labels = []
    if mean_cross:
        data_to_plot.append(mean_cross)
        labels.append("Cross\n(protein-ligand)")
    if mean_lig:
        data_to_plot.append(mean_lig)
        labels.append("Ligand-internal")

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ["coral", "steelblue"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Mean |Shapley value| per edge")
    ax.set_title("Edge importance by type across all explained samples")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_samples(dataset, y_pred_pk, ids_all, n_samples, affinity_set="all"):
    """
    Select samples for XAI explanation.
    affinity_set: 'all' | 'high' | 'low'

    Returns list of (graph_index, graph_object, pred_pk).
    """
    # Build id → dataset index mapping
    id_to_idx = {}
    for i in range(len(dataset)):
        g = dataset[i]
        id_to_idx[g.id] = i

    pairs = list(zip(ids_all, y_pred_pk.tolist()))

    if affinity_set == "high":
        pairs = sorted(pairs, key=lambda x: -x[1])
    elif affinity_set == "low":
        pairs = sorted(pairs, key=lambda x: x[1])
    else:
        rng = default_rng(42)
        idx_arr = np.arange(len(pairs))
        rng.shuffle(idx_arr)
        pairs = [pairs[i] for i in idx_arr]

    selected = []
    for sid, pred in pairs:
        if sid not in id_to_idx:
            continue
        g_idx = id_to_idx[sid]
        g = dataset[g_idx]
        selected.append((g_idx, g, pred))
        if len(selected) >= n_samples:
            break

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",
                   default=os.path.join(os.path.dirname(__file__),
                                        "../../02_data/GEMS_pytorch_datasets/00AEPL_casf2016_indep.pt"))
    p.add_argument("--model_path",
                   default=os.path.join(os.path.dirname(__file__), "GEMS/model"))
    p.add_argument("--n_samples",  type=int, default=30,
                   help="Number of graphs to explain with EdgeSHAPer")
    p.add_argument("--M",          type=int, default=50,
                   help="Monte Carlo sampling steps per edge")
    p.add_argument("--top_k",      type=int, default=10,
                   help="Top-k edges to analyse for physicochemical focus")
    p.add_argument("--affinity_set", choices=["all", "high", "low"], default="all")
    p.add_argument("--output_dir",
                   default=os.path.join(os.path.dirname(__file__),
                                        "results/gems_xai"))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--skip_xai",   action="store_true",
                   help="Run inference only, skip EdgeSHAPer (useful for quick testing)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ Device
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"Device: {device}")

    # ---------------------------------------------------------------- Dataset
    print(f"Loading dataset: {args.dataset_path}")
    dataset = torch.load(args.dataset_path, weights_only=False)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    node_feat_dim = sample.x.shape[1]
    edge_feat_dim = sample.edge_attr.shape[1]
    print(f"Node features: {node_feat_dim}  Edge features: {edge_feat_dim}")

    # Determine model architecture from dataset file name
    dataset_id  = os.path.basename(args.dataset_path)[:6]   # e.g. "00AEPL"
    has_lig_emb = (hasattr(dataset, "ligand_embeddings") and
                   len(dataset.ligand_embeddings) > 0)
    has_prot_emb = (hasattr(dataset, "protein_embeddings") and
                    len(dataset.protein_embeddings) > 0)

    if not has_lig_emb:
        model_arch = "GEMS18e"
        dataset_id = "00AEPL"
    elif not has_prot_emb:
        model_arch = "GEMS18d"
        dataset_id = "00AEPL"
    else:
        model_arch = "GEMS18d"
        # dataset_id already set from filename

    print(f"Model architecture: {model_arch}  Dataset ID: {dataset_id}")

    # ------------------------------------------------------------------ Models
    models = load_ensemble(args.model_path, model_arch, dataset_id,
                           node_feat_dim, edge_feat_dim, device)

    # ------------------------------------------------------------ Inference
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)
    y_true_pk, y_pred_pk, ids_all, metrics, has_labels = run_inference(
        models, loader, device
    )

    # Save inference results
    inf_csv = os.path.join(args.output_dir, "inference_predictions.csv")
    with open(inf_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "y_true_pk", "y_pred_pk"])
        rows = sorted(zip(ids_all, y_true_pk.tolist(), y_pred_pk.tolist()),
                      key=lambda x: x[0])
        w.writerows(rows)
    print(f"Saved inference predictions → {inf_csv}")

    if has_labels and metrics:
        plot_scatter(y_true_pk.numpy(), y_pred_pk.numpy(),
                     metrics["R"], metrics["RMSE"], metrics["R2"],
                     os.path.join(args.output_dir, "inference_scatter.png"))

        metrics_path = os.path.join(args.output_dir, "inference_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics: {metrics}")

    if args.skip_xai:
        print("--skip_xai set. Stopping after inference.")
        return

    # ---------------------------------------------------------- Sample selection
    selected = select_samples(dataset, y_pred_pk, ids_all,
                              args.n_samples, args.affinity_set)
    print(f"Selected {len(selected)} samples for XAI ({args.affinity_set} affinity)")

    # ---------------------------------------------------------- EdgeSHAPer loop
    xai_results_dir = os.path.join(args.output_dir, "edgeshaper")
    os.makedirs(xai_results_dir, exist_ok=True)

    all_stats = []

    for g_idx, graph, pred_pk in tqdm(selected, desc="XAI samples"):
        sid = graph.id
        print(f"\n  Explaining {sid}  (pred pK = {pred_pk:.3f})")

        # --- Node type identification ---
        # n_nodes tensor: [total_nodes, n_lig_nodes, n_prot_nodes]
        n_lig  = int(graph.n_nodes[1].item())

        edge_index = graph.edge_index
        edge_types = classify_edges(edge_index, n_lig)

        n_edges = edge_index.shape[1]
        n_cross = sum(1 for t in edge_types if t == "cross")
        n_lig_e = sum(1 for t in edge_types if t == "lig_internal")
        print(f"    Edges: total={n_edges}  cross={n_cross}  lig_internal={n_lig_e}")

        # --- Run EdgeSHAPer ---
        phi = gems_edgeshaper(models, graph, M=args.M, seed=42, device=device)

        if not phi:
            print(f"    Skipping {sid}: no Shapley values computed")
            continue

        abs_phi = [abs(v) for v in phi]

        # --- Top-k analysis ---
        top_k = min(args.top_k, len(phi))
        top_k_indices = np.argsort(abs_phi)[::-1][:top_k]
        top_k_types   = [edge_types[i] for i in top_k_indices]

        cross_frac = sum(1 for t in top_k_types if t == "cross")  / top_k
        lig_frac   = sum(1 for t in top_k_types if t == "lig_internal") / top_k

        # Mean absolute Shapley per edge type
        cross_abs = [abs_phi[i] for i, t in enumerate(edge_types) if t == "cross"]
        lig_abs   = [abs_phi[i] for i, t in enumerate(edge_types) if t == "lig_internal"]
        mean_abs_cross = float(np.mean(cross_abs)) if cross_abs else 0.0
        mean_abs_lig   = float(np.mean(lig_abs))   if lig_abs   else 0.0

        sample_stats = {
            "id":              sid,
            "pred_pk":         pred_pk,
            "true_pk":         float(unscale(graph.y).item()) if has_labels else None,
            "n_edges":         n_edges,
            "n_cross":         n_cross,
            "n_lig":           n_lig_e,
            "cross_frac_topk": cross_frac,
            "lig_frac_topk":   lig_frac,
            "mean_abs_cross":  mean_abs_cross,
            "mean_abs_lig":    mean_abs_lig,
            "top_k":           top_k,
        }
        all_stats.append(sample_stats)

        # --- Save per-sample results ---
        sample_dir = os.path.join(xai_results_dir, sid)
        os.makedirs(sample_dir, exist_ok=True)

        # Shapley values CSV
        edge_csv = os.path.join(sample_dir, f"{sid}_edge_shapley.csv")
        with open(edge_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["edge_idx", "src", "dst", "edge_type",
                        "shapley_value", "abs_shapley"])
            src = edge_index[0].cpu().numpy()
            dst = edge_index[1].cpu().numpy()
            for i, (phi_val, et) in enumerate(zip(phi, edge_types)):
                w.writerow([i, src[i], dst[i], et,
                            f"{phi_val:.6f}", f"{abs_phi[i]:.6f}"])

        # Per-sample statistics text
        stats_txt = os.path.join(sample_dir, f"{sid}_xai_stats.txt")
        with open(stats_txt, "w") as f:
            f.write(f"Complex: {sid}\n")
            f.write(f"Predicted pK: {pred_pk:.3f}\n")
            if has_labels:
                f.write(f"True pK:      {sample_stats['true_pk']:.3f}\n")
            f.write(f"\nEdge counts:\n")
            f.write(f"  Total edges:    {n_edges}\n")
            f.write(f"  Cross edges:    {n_cross} ({n_cross/n_edges*100:.1f}%)\n")
            f.write(f"  Lig-internal:   {n_lig_e} ({n_lig_e/n_edges*100:.1f}%)\n")
            f.write(f"\nTop-{top_k} edge analysis (by |Shapley value|):\n")
            f.write(f"  Cross edges in top-{top_k}:         {cross_frac*100:.1f}%\n")
            f.write(f"  Ligand-internal in top-{top_k}:     {lig_frac*100:.1f}%\n")
            f.write(f"\nMean |Shapley| per edge type:\n")
            f.write(f"  Cross edges:    {mean_abs_cross:.6f}\n")
            f.write(f"  Lig-internal:   {mean_abs_lig:.6f}\n")

        # Per-sample Shapley distribution plot
        plot_shapley_histogram(
            phi, edge_types, sid,
            os.path.join(sample_dir, f"{sid}_shapley_distribution.png")
        )

        print(f"    top-{top_k}: cross={cross_frac*100:.1f}%  lig={lig_frac*100:.1f}%  "
              f"mean_abs_cross={mean_abs_cross:.4f}  mean_abs_lig={mean_abs_lig:.4f}")

    if not all_stats:
        print("No XAI results generated.")
        return

    # ---------------------------------------------------------- Aggregate analysis
    print("\n=== Aggregate XAI Analysis ===")
    mean_cross_frac = np.mean([s["cross_frac_topk"] for s in all_stats])
    mean_lig_frac   = np.mean([s["lig_frac_topk"]   for s in all_stats])
    mean_abs_cross  = np.mean([s["mean_abs_cross"]   for s in all_stats if s["n_cross"] > 0])
    mean_abs_lig    = np.mean([s["mean_abs_lig"]     for s in all_stats if s["n_lig"]   > 0])
    print(f"  Mean fraction cross in top-{args.top_k}: {mean_cross_frac:.3f}")
    print(f"  Mean fraction lig   in top-{args.top_k}: {mean_lig_frac:.3f}")
    print(f"  Mean |Shapley| cross:  {mean_abs_cross:.6f}")
    print(f"  Mean |Shapley| lig:    {mean_abs_lig:.6f}")

    # Cross-edge dominance score: fraction of samples where cross > lig in top-k
    dominance = sum(1 for s in all_stats if s["cross_frac_topk"] > s["lig_frac_topk"])
    print(f"  Cross-edge dominated ({len(all_stats)} samples): "
          f"{dominance}/{len(all_stats)} = {dominance/len(all_stats)*100:.1f}%")

    if mean_cross_frac > mean_lig_frac:
        print("\n  INTERPRETATION: Cross (protein-ligand) edges dominate → "
              "model focuses on real binding interface interactions.")
    else:
        print("\n  INTERPRETATION: Ligand-internal edges dominate → "
              "potential statistical bias; model may not focus on binding interactions.")

    # Save aggregate stats
    agg_csv = os.path.join(args.output_dir, "aggregate_xai_stats.csv")
    with open(agg_csv, "w", newline="") as f:
        w = csv.writer(f)
        if all_stats:
            w.writerow(all_stats[0].keys())
            for s in all_stats:
                w.writerow(s.values())
    print(f"Saved aggregate stats → {agg_csv}")

    agg_json = os.path.join(args.output_dir, "aggregate_xai_summary.json")
    with open(agg_json, "w") as f:
        json.dump({
            "n_samples_explained":      len(all_stats),
            "top_k":                    args.top_k,
            "M_monte_carlo":            args.M,
            "affinity_set":             args.affinity_set,
            "mean_cross_frac_topk":     float(mean_cross_frac),
            "mean_lig_frac_topk":       float(mean_lig_frac),
            "mean_abs_shapley_cross":   float(mean_abs_cross),
            "mean_abs_shapley_lig":     float(mean_abs_lig),
            "cross_dominant_fraction":  float(dominance / len(all_stats)),
        }, f, indent=2)

    # Aggregate plots
    plot_edge_type_fractions(
        all_stats, args.top_k,
        os.path.join(args.output_dir, "aggregate_topk_edge_types.png")
    )
    plot_mean_shapley_by_type(
        all_stats,
        os.path.join(args.output_dir, "aggregate_mean_shapley_by_type.png")
    )

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
