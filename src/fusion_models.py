"""
src/intern4_fusion/fusion_models.py
=====================================
Implements all required fusion strategies (Task 4.2 + 4.3):
  1. Early Fusion       — concatenate raw features -> single classifier
  2. Intermediate Fusion— concatenate latent vectors -> classifier
  3. Late Fusion         — weighted average of probability vectors
  4. Cross-Attention     — multi-head attention treating each modality
                           as a token (primary method per spec)

All four produce a (N, 3) probability vector so they can be stacked
by the XGBoost meta-learner in Task 4.4.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# STRATEGY 1 — EARLY FUSION
# Concatenate raw feature vectors, train a single shallow classifier.
# ═══════════════════════════════════════════════════════════

class EarlyFusionModel(nn.Module):
    """
    Concatenates raw modality vectors at the input level and learns
    a single joint representation. This is the classical "fuse first,
    learn once" baseline.

    Input: base_feat (B, d_base) + ext_feat (B, d_ext)  ->  concat (B, d_base+d_ext)
    """
    def __init__(self, d_base: int, d_ext: int, n_classes: int = 3, hidden: int = 128):
        super().__init__()
        d_in = d_base + d_ext
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, base_feat: torch.Tensor, ext_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([base_feat, ext_feat], dim=-1)
        return self.net(x)   # logits (B, n_classes)


# ═══════════════════════════════════════════════════════════
# STRATEGY 2 — INTERMEDIATE FUSION
# Each modality first goes through its own projection head to a
# shared latent size, THEN gets concatenated and classified.
# This differs from Early Fusion by learning modality-specific
# representations before fusing (matches the project's stated
# "penultimate layer" definition).
# ═══════════════════════════════════════════════════════════

class IntermediateFusionModel(nn.Module):
    def __init__(self, d_base: int, d_ext: int, latent_dim: int = 256,
                n_classes: int = 3):
        super().__init__()
        self.base_proj = nn.Sequential(
            nn.Linear(d_base, latent_dim), nn.ReLU(), nn.Dropout(0.3)
        )
        self.ext_proj = nn.Sequential(
            nn.Linear(d_ext, latent_dim), nn.ReLU(), nn.Dropout(0.3)
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, base_feat: torch.Tensor, ext_feat: torch.Tensor):
        b = self.base_proj(base_feat)     # (B, latent_dim)
        e = self.ext_proj(ext_feat)       # (B, latent_dim)
        fused = torch.cat([b, e], dim=-1)  # (B, 2*latent_dim)
        return self.head(fused), (b, e)    # logits, latents (for downstream use)


# ═══════════════════════════════════════════════════════════
# STRATEGY 3 — LATE FUSION
# Pure score-level weighted average. No learnable params.
# ═══════════════════════════════════════════════════════════

def late_fusion(base_probs: np.ndarray, ext_probs: np.ndarray,
                alpha: float = 0.5) -> tuple:
    assert base_probs.shape == ext_probs.shape
    fused = alpha * base_probs + (1.0 - alpha) * ext_probs
    return fused, fused.argmax(axis=1)


# ═══════════════════════════════════════════════════════════
# STRATEGY 4 — CROSS-ATTENTION FUSION (primary method, Task 4.3)
# Treats Base and Extended as two tokens in a sequence and lets
# multi-head self-attention mix information between them.
# Output: unified latent representation (batch, 768) as specified.
# ═══════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Modality_i = CrossAttn(Modality_i, Concat(Modality_j != i))

    With 2 modalities (Base, Extended), this reduces to standard
    bidirectional multi-head self-attention over a 2-token sequence,
    which is the practical implementation of the spec's formula for
    the 2-modality case.

    Output dim is forced to 768 per Task 4.3's explicit requirement.
    """
    def __init__(self, d_base: int, d_ext: int,
                 d_model: int = 384, n_heads: int = 8,
                 n_classes: int = 3, output_dim: int = 768):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.base_proj = nn.Linear(d_base, d_model)
        self.ext_proj = nn.Linear(d_ext, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        # 2 tokens * d_model -> output_dim (768 as required)
        self.to_unified = nn.Sequential(
            nn.Linear(d_model * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, base_feat: torch.Tensor, ext_feat: torch.Tensor):
        b = self.base_proj(base_feat)        # (B, d_model)
        e = self.ext_proj(ext_feat)          # (B, d_model)

        seq = torch.stack([b, e], dim=1)     # (B, 2, d_model)  — 2 tokens
        attn_out, attn_weights = self.attn(seq, seq, seq)
        attn_out = self.norm(attn_out + seq)  # residual

        flat = attn_out.flatten(1)            # (B, 2*d_model)
        unified = self.to_unified(flat)       # (B, 768)
        logits = self.cls_head(unified)       # (B, n_classes)

        return logits, unified, attn_weights


# ═══════════════════════════════════════════════════════════
# Training helper shared by all three learnable models
# ═══════════════════════════════════════════════════════════

def train_torch_classifier(model: nn.Module,
                           base_train: torch.Tensor, ext_train: torch.Tensor,
                           y_train: torch.Tensor,
                           base_val: torch.Tensor, ext_val: torch.Tensor,
                           y_val: torch.Tensor,
                           epochs: int = 60, lr: float = 1e-3,
                           class_weights: torch.Tensor = None,
                           model_type: str = "early") -> nn.Module:
    """
    Generic training loop for EarlyFusionModel / IntermediateFusionModel /
    CrossAttentionFusion. model_type controls how forward() output is unpacked.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_val_loss = float("inf")
    best_state = None
    patience, patience_ctr = 10, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(base_train, ext_train)
        logits = out[0] if isinstance(out, tuple) else out
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(base_val, ext_val)
            logits_val = out_val[0] if isinstance(out_val, tuple) else out_val
            val_loss = criterion(logits_val, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def get_probs(model: nn.Module, base_feat: torch.Tensor,
              ext_feat: torch.Tensor) -> np.ndarray:
    """Run inference and return softmax probabilities as numpy array."""
    model.eval()
    with torch.no_grad():
        out = model(base_feat, ext_feat)
        logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits, dim=-1)
    return probs.numpy()
