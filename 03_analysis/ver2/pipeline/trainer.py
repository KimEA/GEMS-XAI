"""
Module 2: Trainer
==================
SimpleGCN 모델 정의 및 학습 루프.

비교 모델 구조:
    - GEMS18d:   NodeTransform + 2× MetaLayer(EdgeModel+NodeModel+GlobalModel) + FC
                 → 복잡한 메시지 패싱, lig_emb 초기 전역 특징으로 활용
    - SimpleGCN: NodeTransform + 2× GATv2Conv + GlobalAddPool + concat(lig_emb) + FC
                 → MetaLayer 없는 단순한 구조 (GEMS 대비 표현력 하위 모델)

두 모델이 동일한 B6AEPL 그래프 포맷(graphbatch)을 입력으로 사용하여
공정한 비교 가능.

학습 설정 예시:
    - PDBbind  버전: B6AEPL_train_pdbbind.pt  전체 사용
    - CleanSplit 버전: B6AEPL_train_cleansplit.pt 의 80% 사용
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool

# ─── 손실 함수 ────────────────────────────────────────────────────────────────

class RMSELoss(nn.Module):
    """Root Mean Square Error Loss."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, target))


# ─── SimpleGCN 모델 ───────────────────────────────────────────────────────────

class SimpleGCN(nn.Module):
    """
    GEMS 그래프 포맷에서 작동하는 단순 GCN 기반 결합 친화도 예측 모델.

    GEMS18d와 차이점:
        - MetaLayer(Edge/Node/Global 업데이트)를 사용하지 않음
        - 단순 GATv2Conv + GlobalAddPool 구조
        - lig_emb를 노드 레이어 이후 concat하여 전역 문맥으로만 사용

    Args:
        in_channels:  노드 특징 차원 (B6AEPL에서 1148)
        edge_dim:     엣지 특징 차원 (B6AEPL에서 20)
        hidden:       내부 은닉 차원 (기본 256)
        n_heads:      GATv2Conv의 어텐션 헤드 수 (기본 4)
        n_layers:     GATv2Conv 레이어 수 (기본 2)
        dropout:      드롭아웃 비율 (기본 0.1)
        lig_emb_dim:  리간드 임베딩 차원 (ChemBERTa-77M = 384)
    """

    def __init__(
        self,
        in_channels:  int,
        edge_dim:     int,
        hidden:       int = 256,
        n_heads:      int = 4,
        n_layers:     int = 2,
        dropout:      float = 0.1,
        lig_emb_dim:  int = 384,
    ):
        super().__init__()
        assert hidden % n_heads == 0, "hidden은 n_heads의 배수여야 합니다"
        out_per_head = hidden // n_heads

        # ─ 1. 노드 특징 변환 MLP (고차원 → 은닉 차원)
        # B6AEPL의 1148-dim → 256-dim으로 압축
        self.node_transform = nn.Sequential(
            nn.Linear(in_channels, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

        # ─ 2. GATv2Conv 레이어들 (엣지 특징 포함)
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                GATv2Conv(
                    hidden,
                    out_per_head,
                    edge_dim=edge_dim,
                    heads=n_heads,
                    dropout=dropout,
                    add_self_loops=False,  # GEMS 그래프는 별도 self-loop 처리
                )
            )
            self.bns.append(BatchNorm1d(hidden))

        # ─ 3. 잔차 연결(Residual) 조정용 프로젝션
        self.res_proj = nn.Linear(hidden, hidden)

        # ─ 4. 전역 풀링 + 리간드 임베딩 결합 후 예측
        # GlobalAddPool 결과(hidden) + lig_emb(384) → FC layers
        concat_dim = hidden + lig_emb_dim
        self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = nn.Linear(concat_dim, 128)
        self.fc2 = nn.Linear(128, 1)

        # lig_emb가 없는 경우를 위한 제로 벡터 (예: 00AEPL 데이터셋)
        self.lig_emb_dim = lig_emb_dim

    def forward(self, graphbatch):
        """
        Args:
            graphbatch: PyG Batch 객체 (DataLoader에서 배치된 그래프)
                - graphbatch.x:          [N_total, in_channels]
                - graphbatch.edge_index: [2, N_edges]
                - graphbatch.edge_attr:  [N_edges, edge_dim]
                - graphbatch.lig_emb:    [B, lig_emb_dim] 또는 None
                - graphbatch.batch:      [N_total]  (각 노드의 그래프 인덱스)
        Returns:
            out: [B, 1]  (배치 내 각 그래프의 예측값, 정규화 스케일)
        """
        x          = graphbatch.x
        edge_index = graphbatch.edge_index
        edge_attr  = graphbatch.edge_attr
        batch      = graphbatch.batch

        # ─ Step 1: 노드 특징 변환
        x = self.node_transform(x)                      # [N, hidden]

        # ─ Step 2: GATv2Conv 레이어 통과 (잔차 연결 포함)
        for conv, bn in zip(self.convs, self.bns):
            x_res = self.res_proj(x)                    # 잔차용 프로젝션
            x = F.relu(conv(x, edge_index, edge_attr))  # [N, hidden]
            x = bn(x)
            x = x + x_res                              # 잔차 연결

        # ─ Step 3: 그래프 수준 전역 표현
        x_global = global_add_pool(x, batch)            # [B, hidden]

        # ─ Step 4: 리간드 임베딩 결합
        if hasattr(graphbatch, "lig_emb") and graphbatch.lig_emb is not None:
            lig_emb = graphbatch.lig_emb
            # DataLoader 배치 후: [B, 1, 384] → [B, 384] 처리
            if lig_emb.dim() == 3:
                lig_emb = lig_emb.squeeze(1)
            x_global = torch.cat([x_global, lig_emb], dim=1)  # [B, hidden+384]
        else:
            # lig_emb 없는 경우 zero padding
            zeros = torch.zeros(
                x_global.shape[0], self.lig_emb_dim,
                device=x_global.device, dtype=x_global.dtype
            )
            x_global = torch.cat([x_global, zeros], dim=1)

        # ─ Step 5: 예측 FC 레이어
        out = self.dropout_layer(x_global)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# ─── 모델 저장 / 로드 ─────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_metrics, save_path):
    """
    모델 체크포인트 저장 (가중치 + 메타정보).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "model_config": {
            "in_channels":  model.node_transform[0].in_features,
            "edge_dim":     model.convs[0].edge_dim,
            "hidden":       model.node_transform[-1].out_features,
            "n_layers":     len(model.convs),
            "lig_emb_dim":  model.lig_emb_dim,
        }
    }, save_path)
    print(f"[Trainer] 체크포인트 저장: {save_path}")


def load_gcn_checkpoint(path: str, device) -> SimpleGCN:
    """
    저장된 체크포인트에서 SimpleGCN 모델 복원.
    """
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    cfg    = ckpt["model_config"]
    model  = SimpleGCN(
        in_channels = cfg["in_channels"],
        edge_dim    = cfg["edge_dim"],
        hidden      = cfg["hidden"],
        n_layers    = cfg["n_layers"],
        lig_emb_dim = cfg["lig_emb_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Trainer] GCN 모델 로드: {path}  (epoch {ckpt['epoch']})")
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
    model_name: str = "gcn",
) -> dict:
    """
    SimpleGCN 전체 학습 루프 (조기 종료 포함).

    Args:
        model:        SimpleGCN 인스턴스
        train_loader: 학습 DataLoader
        val_loader:   검증 DataLoader
        config:       학습 설정 딕셔너리 (아래 기본값 참고)
        device:       torch.device
        save_dir:     체크포인트 저장 디렉터리 (None이면 저장 안 함)
        model_name:   저장 파일명 접두어

    config 키 목록:
        - lr:           학습률 (기본 1e-4)
        - weight_decay: 가중치 감쇠 (기본 1e-5)
        - epochs:       최대 에포크 수 (기본 300)
        - patience:     조기 종료 인내값 (기본 30)
        - scheduler:    LR 스케줄러 사용 여부 (기본 True)

    Returns:
        history: {"train_loss": [...], "val_loss": [...], "best_epoch": int}
    """
    # 기본 학습 설정
    lr           = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-5)
    epochs       = config.get("epochs", 300)
    patience     = config.get("patience", 30)
    use_scheduler = config.get("scheduler", True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = RMSELoss()

    # Cosine Annealing LR 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    ) if use_scheduler else None

    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_val  = float("inf")
    patience_cnt = 0
    best_state   = None

    print(f"\n[Trainer] === {model_name} 학습 시작 ===")
    print(f"          epochs={epochs}, lr={lr}, patience={patience}")
    print(f"          학습 배치 수={len(train_loader)}, 검증 배치 수={len(val_loader)}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = _validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # 모델 저장 조건: 검증 손실 개선
        if val_loss < best_val:
            best_val         = val_loss
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            patience_cnt     = 0

            if save_dir:
                ckpt_path = os.path.join(save_dir, f"{model_name}_best.pt")
                save_checkpoint(
                    model, optimizer, epoch,
                    {"val_rmse": val_loss},
                    ckpt_path
                )
        else:
            patience_cnt += 1

        # 에포크 로그 (10 에포크마다 출력)
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"train_RMSE={train_loss:.4f} | val_RMSE={val_loss:.4f} | "
                  f"best={best_val:.4f} (epoch {history['best_epoch']}) | "
                  f"{elapsed:.0f}s")

        # 조기 종료
        if patience_cnt >= patience:
            print(f"  [조기 종료] {patience}에포크 동안 개선 없음. 종료.")
            break

    # 최적 가중치 복원
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

    elapsed_total = time.time() - t0
    print(f"[Trainer] 학습 완료: best_epoch={history['best_epoch']}, "
          f"best_val_RMSE={best_val:.4f}, 총 소요={elapsed_total:.0f}s\n")

    # 학습 이력 저장
    if save_dir:
        hist_path = os.path.join(save_dir, f"{model_name}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    return history


# ─── 빠른 테스트용 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # B6AEPL 차원으로 SimpleGCN 인스턴스화 및 순전파 테스트
    from torch_geometric.data import Data, Batch
    device = torch.device("cpu")

    model = SimpleGCN(in_channels=1148, edge_dim=20, hidden=256, lig_emb_dim=384).to(device)
    print(f"SimpleGCN 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 더미 배치 생성
    dummy = Data(
        x          = torch.randn(50, 1148),
        edge_index = torch.randint(0, 50, (2, 400)),
        edge_attr  = torch.randn(400, 20),
        y          = torch.tensor([0.5]),
        n_nodes    = torch.tensor([50, 26, 24]),
        lig_emb    = torch.randn(1, 384),
    )
    batch = Batch.from_data_list([dummy, dummy])
    out = model(batch)
    print(f"출력 shape: {out.shape}")  # [2, 1] 기대
