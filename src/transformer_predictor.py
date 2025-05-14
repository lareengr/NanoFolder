import torch
import torch.nn as nn

class TransformerDistancePredictor(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 1280,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # our transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,   # so input shape is [B, L, D]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # a simple MLP head that takes [2*D]→512→1 per residue‐pair
        self.dist_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts an L×L distance map from per‐residue embeddings.

        Args:
          x: either
             - [L, D]   (a single sequence, un‐batched), or
             - [1, L, D] (a single sequence, batched).
        Returns:
          dist: [L, L] matrix of predicted pairwise distances.
        """
        # --- 1) normalize input to [1, L, D] ---
        if x.dim() == 2:
            # un‐batched → add batch dim
            x_in = x.unsqueeze(0)      # [1, L, D]
        elif x.dim() == 3 and x.size(0) == 1:
            # already [1, L, D]
            x_in = x
        else:
            raise ValueError(f"Expected input shape [L,D] or [1,L,D], got {tuple(x.shape)}")

        # --- 2) encode with transformer --- 
        # out: [1, L, D]
        x_enc = self.encoder(x_in)

        # drop batch dim → [L, D]
        x_enc = x_enc.squeeze(0)

        L, D = x_enc.shape

        # --- 3) build pairwise representation ---
        # xi[i,j,:] = x_enc[j,:] along j, so we unsqueeze at dim=1 and expand
        # xj[i,j,:] = x_enc[i,:] along i, unsqueeze at dim=0 and expand
        xi = x_enc.unsqueeze(1).expand(L, L, D)  # [L, L, D]
        xj = x_enc.unsqueeze(0).expand(L, L, D)  # [L, L, D]

        pairwise = torch.cat([xi, xj], dim=-1)   # [L, L, 2D]

        # --- 4) predict distances ---
        # head: [L, L, 1] → squeeze → [L, L]
        dist = self.dist_head(pairwise).squeeze(-1)

        return dist
