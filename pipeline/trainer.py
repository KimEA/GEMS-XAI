"""
Module 2: Trainer
==================
모델 정의 및 학습 루프.

비교 모델 구조:
    - GEMS18d:  NodeTransform + 2× MetaLayer(EdgeModel+NodeModel+GlobalModel) + FC
                → 복잡한 메시지 패싱, lig_emb 초기 전역 특징, edge_attr 활용
    - GC_GNN:   GraphConv × 7 (max 집계) + GlobalAddPool + FC  (논문 원본)
                → edge_attr 미사용, 그래프 위상(topology)만 학습
                → 출처: protein-ligand-GNN/src/utils.py (Andrea Mastropietro)
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GraphConv, global_add_pool

# GEMS18d 임포트
GEMS_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "GEMS")
sys.path.insert(0, GEMS_MODEL_DIR)
from model.GEMS18 import GEMS18d

# ─── 손실 함수 ────────────────────────────────────────────────────────────────

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, target))


# ─── GC_GNN 모델 (논문 원본 코드) ────────────────────────────────────────────
# 출처: protein-ligand-GNN/src/utils.py  (Andrea Mastropietro)
# 원본 코드를 수정 없이 그대로 사용.

class GC_GNN(torch.nn.Module):
    def __init__(self, node_features_dim, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GraphConv(node_features_dim, hidden_channels, aggr='max')
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr='max')
        self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr='max')
        self.conv4 = GraphConv(hidden_channels, hidden_channels, aggr='max')
        self.conv5 = GraphConv(hidden_channels, hidden_channels, aggr='max')
        self.conv6 = GraphConv(hidden_channels, hidden_channels, aggr='max')
        self.conv7 = GraphConv(hidden_channels, hidden_channels, aggr='max')
        self.lin = geom_nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv4(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv5(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv6(x, edge_index, edge_weight=edge_weight))
        x = self.conv7(x, edge_index, edge_weight=edge_weight)
        x = global_add_pool(x, batch)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        return x


class GCNNWrapper(nn.Module):
    """
    논문 원본 GC_GNN을 GEMS graphbatch 인터페이스로 래핑.

    GC_GNN.forward(x, edge_index, batch, edge_weight) →
    GCNNWrapper.forward(graphbatch)

    edge_weight = edge_attr[:, 3]  (거리/10)
        - 공유결합:   결합 길이 / 10
        - 비공유결합: 리간드 원자 → 단백질 Cα 거리 / 10
    → 거리 기반 상호작용 강도 학습 가능 (원본 논문 의도 반영).
    """

    def __init__(self, inner: GC_GNN):
        super().__init__()
        self.inner = inner

    def forward(self, graphbatch):
        edge_weight = None
        if graphbatch.edge_attr is not None and graphbatch.edge_attr.shape[1] > 3:
            edge_weight = graphbatch.edge_attr[:, 3]  # 거리/10

        return self.inner(
            graphbatch.x.float(),
            graphbatch.edge_index,
            graphbatch.batch,
            edge_weight=edge_weight,
        )


def build_gcngnn(node_feat_dim: int, hidden: int = 256, device=None) -> GCNNWrapper:
    """GC_GNN(논문 원본) + GCNNWrapper 생성."""
    inner   = GC_GNN(node_features_dim=node_feat_dim,
                     hidden_channels=hidden, num_classes=1).float()
    wrapper = GCNNWrapper(inner)
    if device is not None:
        wrapper = wrapper.to(device)
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f"[Trainer] GC_GNN 초기화: {n_params:,} 파라미터 "
          f"(node_feat={node_feat_dim}, hidden={hidden}, layers=7)")
    return wrapper


def save_gcngnn_checkpoint(
    model, optimizer, epoch, val_metrics, save_path,
    node_feat_dim=None, hidden=None,
):
    """GCNNWrapper 체크포인트 저장."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    inner = model.inner if isinstance(model, GCNNWrapper) else model
    _nfd  = node_feat_dim or inner.conv1.lin_root.weight.shape[1]
    _hid  = hidden        or inner.conv1.lin_root.weight.shape[0]
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "model_config": {
            "model_class":       "GC_GNN",
            "node_features_dim": _nfd,
            "hidden_channels":   _hid,
            "num_classes":       1,
        }
    }, save_path)
    print(f"[Trainer] GC_GNN 체크포인트 저장: {save_path}")


def load_gcngnn_checkpoint(path: str, device) -> GCNNWrapper:
    """저장된 체크포인트에서 GCNNWrapper(GC_GNN) 복원."""
    ckpt    = torch.load(path, map_location=device, weights_only=False)
    cfg     = ckpt["model_config"]
    inner   = GC_GNN(node_features_dim=cfg["node_features_dim"],
                     hidden_channels=cfg["hidden_channels"],
                     num_classes=cfg["num_classes"]).float()
    wrapper = GCNNWrapper(inner)
    wrapper.load_state_dict(ckpt["model_state"])
    wrapper = wrapper.to(device)
    wrapper.eval()
    print(f"[Trainer] GC_GNN 모델 로드: {path}  (epoch {ckpt['epoch']})")
    return wrapper


# ─── GEMS18d 빌드 / 저장 / 로드 ──────────────────────────────────────────────

def build_gems18d(
    node_feat_dim: int,
    edge_feat_dim: int,
    dropout_prob: float = 0.5,
    conv_dropout_prob: float = 0.0,
    device=None,
) -> GEMS18d:
    """GEMS18d 인스턴스 생성 (처음부터 학습용)."""
    model = GEMS18d(
        dropout_prob      = dropout_prob,
        in_channels       = node_feat_dim,
        edge_dim          = edge_feat_dim,
        conv_dropout_prob = conv_dropout_prob,
    ).float()
    if device is not None:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Trainer] GEMS18d 초기화: {n_params:,} 파라미터 "
          f"(node_feat={node_feat_dim}, edge_feat={edge_feat_dim})")
    return model


def save_gems_checkpoint(
    model,
    optimizer,
    epoch: int,
    val_metrics: dict,
    save_path: str,
    node_feat_dim: int = None,
    edge_feat_dim: int = None,
):
    """GEMS18d 체크포인트 저장."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "model_config": {
            "model_class":       "GEMS18d",
            "node_feat_dim":     node_feat_dim or model.NodeTransform.mlp[0].in_features,
            "edge_feat_dim":     edge_feat_dim or model.layer1.edge_model.edge_mlp[0].in_features - 128,
            "dropout_prob":      model.dropout_layer.p,
            "conv_dropout_prob": 0,
        }
    }, save_path)
    print(f"[Trainer] GEMS 체크포인트 저장: {save_path}")


def load_gems_checkpoint(path: str, device) -> GEMS18d:
    """저장된 체크포인트에서 GEMS18d 모델 복원."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]
    model = GEMS18d(
        dropout_prob      = cfg.get("dropout_prob", 0.5),
        in_channels       = cfg["node_feat_dim"],
        edge_dim          = cfg["edge_feat_dim"],
        conv_dropout_prob = cfg.get("conv_dropout_prob", 0),
    ).float().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Trainer] GEMS18d 모델 로드: {path}  (epoch {ckpt['epoch']})")
    return model


# ─── 학습 루프 ────────────────────────────────────────────────────────────────

def _train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """에포크 한 번 학습 수행. 평균 RMSE 반환."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch).view(-1)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def _validate(model, loader, criterion, device) -> float:
    """검증 셋 평가. 평균 RMSE 반환."""
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred  = model(batch).view(-1)
        total_loss += criterion(pred, batch.y).item()
    return total_loss / len(loader)


def train_model(
    model,
    train_loader,
    val_loader,
    config: dict,
    device,
    save_dir: str = None,
    model_name: str = "model",
    save_fn=None,
) -> dict:
    """
    전체 학습 루프 (조기 종료 포함). GEMS18d / GCNNWrapper 모두 지원.

    config 키:
        lr, weight_decay, epochs, patience, es_tolerance, scheduler,
        optimizer ("adam" | "sgd"), momentum (SGD 전용, 기본 0.9)

    es_tolerance: early stopping 최소 개선량 (기본 0.0 = 어떤 개선도 허용)
        GEMS 원논문: 0.01
        val_loss < best_val - es_tolerance 일 때만 개선으로 인정
    """
    lr            = config.get("lr", 1e-4)
    weight_decay  = config.get("weight_decay", 1e-5)
    epochs        = config.get("epochs", 300)
    patience      = config.get("patience", 30)
    es_tolerance  = config.get("es_tolerance", 0.0)
    use_scheduler = config.get("scheduler", True)
    opt_type      = config.get("optimizer", "adam").lower()

    if opt_type == "sgd":
        momentum  = config.get("momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            momentum=momentum,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = RMSELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    ) if use_scheduler else None

    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_val     = float("inf")
    patience_cnt = 0
    best_state   = None

    print(f"\n[Trainer] === {model_name} 학습 시작 ===")
    print(f"          epochs={epochs}, lr={lr}, patience={patience}, es_tol={es_tolerance}")
    print(f"          학습 배치={len(train_loader)}, 검증 배치={len(val_loader)}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = _validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - es_tolerance:
            best_val              = val_loss
            best_state            = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            patience_cnt          = 0

            if save_dir:
                ckpt_path = os.path.join(save_dir, f"{model_name}_best.pt")
                _save_fn  = save_fn if save_fn is not None else (lambda *a, **kw: None)
                _save_fn(model, optimizer, epoch, {"val_rmse": val_loss}, ckpt_path)
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | "
                  f"best={best_val:.4f} (ep {history['best_epoch']}) | {elapsed:.0f}s")

        if patience_cnt >= patience:
            print(f"  [조기 종료] {patience}에포크 개선 없음.")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

    elapsed_total = time.time() - t0
    print(f"[Trainer] 완료: best_epoch={history['best_epoch']}, "
          f"best_val_RMSE={best_val:.4f}, 소요={elapsed_total:.0f}s\n")

    if save_dir:
        with open(os.path.join(save_dir, f"{model_name}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    return history
