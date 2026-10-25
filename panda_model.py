"""Self-contained PANDA architecture for the public checkpoint.

This module defines the exact inference architecture:

    TraceShrinkageBackoff(JointAdaptivePairwiseEstimator(hidden=48))

It has no experiment-generation or training-loop dependencies.  Its parameter
names and forward computation are checkpoint-compatible with
``checkpoints/panda_dimension_generalization.pt``.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional


def assemble_scatter(lowrank: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    """Return the low-rank-plus-diagonal covariance matrix."""
    return lowrank + torch.diag_embed(diagonal)


def _validate_inputs(
    sample_scatter: torch.Tensor,
    ranks: torch.Tensor,
    diagonal: torch.Tensor | None = None,
) -> None:
    if sample_scatter.ndim != 3 or sample_scatter.shape[-1] != sample_scatter.shape[-2]:
        raise ValueError("sample_scatter must have shape (batch, p, p)")
    batch_size, p, _ = sample_scatter.shape
    if ranks.shape != (batch_size,):
        raise ValueError("ranks must have shape (batch,)")
    if bool(((ranks < 1) | (ranks >= p)).any()):
        raise ValueError("each rank must be between one and p - 1")
    if diagonal is not None and diagonal.shape != (batch_size, p):
        raise ValueError("diagonal must have shape (batch, p)")


def conditional_lowrank_with_spectrum(
    sample_scatter: torch.Tensor,
    diagonal: torch.Tensor,
    ranks: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One EVD-based, rank-constrained low-rank reconstruction."""
    _validate_inputs(sample_scatter, ranks, diagonal)
    p = sample_scatter.shape[-1]
    root = diagonal.clamp_min(eps).sqrt()
    whitened = sample_scatter / (root.unsqueeze(-1) * root.unsqueeze(-2))
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
    positions = torch.arange(p, device=sample_scatter.device)
    signal_mask = positions.unsqueeze(0) >= (p - ranks).unsqueeze(1)
    strengths = (eigenvalues - 1.0).clamp_min(0.0).sqrt()
    strengths = strengths * signal_mask.to(sample_scatter.dtype)
    loading = root.unsqueeze(-1) * eigenvectors * strengths.unsqueeze(-2)
    lowrank = loading @ loading.transpose(-2, -1)
    return lowrank, eigenvalues, eigenvectors


def fad_projection(
    sample_scatter: torch.Tensor,
    diagonal: torch.Tensor,
    ranks: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First analytic projection, returning its residual positive diagonal."""
    lowrank, eigenvalues, eigenvectors = conditional_lowrank_with_spectrum(
        sample_scatter, diagonal, ranks, eps=eps
    )
    projected_diagonal = torch.diagonal(
        sample_scatter - lowrank, dim1=-2, dim2=-1
    ).clamp_min(eps)
    return lowrank, projected_diagonal, eigenvalues, eigenvectors


class PairwiseMessageCorrection(nn.Module):
    """One permutation-equivariant coordinate-wise message-passing expert."""

    edge_size = 3

    def __init__(self, feature_size: int, hidden: int) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.message = nn.Sequential(
            nn.Linear(2 * hidden + self.edge_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        embedding = self.node_encoder(features)
        batch_size, p, hidden = embedding.shape
        receiver = embedding.unsqueeze(2).expand(-1, -1, p, -1)
        sender = embedding.unsqueeze(1).expand(-1, p, -1, -1)
        messages = self.message(torch.cat((receiver, sender, edges), dim=-1))
        off_diagonal = ~torch.eye(p, dtype=torch.bool, device=features.device)
        messages = messages * off_diagonal[None, :, :, None]
        pooled_messages = messages.sum(dim=2) / float(max(p - 1, 1))
        mean = embedding.mean(dim=1, keepdim=True)
        centered_rms = (
            (embedding - mean).square().mean(dim=1, keepdim=True) + 1e-12
        ).sqrt()
        context = torch.cat((mean, centered_rms), dim=-1).expand(batch_size, p, -1)
        decoder_input = torch.cat((embedding, pooled_messages, context), dim=-1)
        return self.decoder(decoder_input)


class TwoExpertPairwiseCorrection(nn.Module):
    """Mixture of two Pairwise experts, gated by five global task features."""

    def __init__(self, feature_size: int, hidden: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList([
            PairwiseMessageCorrection(feature_size, hidden),
            PairwiseMessageCorrection(feature_size, hidden),
        ])
        self.gate = nn.Sequential(
            nn.Linear(5, hidden), nn.Tanh(), nn.Linear(hidden, 2)
        )

    def forward(self, features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        global_features = features[:, 0, 6:11]
        weights = torch.softmax(self.gate(global_features), dim=-1)
        predictions = torch.stack(
            [expert(features, edges).squeeze(-1) for expert in self.experts],
            dim=-1,
        )
        return (predictions * weights[:, None, :]).sum(dim=-1)


class JointAdaptivePairwiseEstimator(nn.Module):
    """Two-EVD structured covariance estimator with a two-expert Pairwise MoE."""

    feature_size = 16

    def __init__(
        self,
        hidden: int = 48,
        eps: float = 1e-6,
        initial_max_log_step: float = 0.5,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.network = TwoExpertPairwiseCorrection(self.feature_size, hidden)
        initial_value = torch.log(torch.expm1(torch.tensor(initial_max_log_step)))
        self.log_step_scale = nn.Parameter(initial_value)

    def _residual_energy(self, whitened_residual: torch.Tensor) -> torch.Tensor:
        return whitened_residual.square().sum(dim=-1).log1p()

    def _base_features(
        self,
        normalized: torch.Tensor,
        diagonal: torch.Tensor,
        lowrank: torch.Tensor,
        projected_diagonal: torch.Tensor,
        eigenvalues: torch.Tensor,
        sample_counts: torch.Tensor,
        ranks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, p, _ = normalized.shape
        profiled_scatter = assemble_scatter(lowrank, diagonal)
        identity = torch.eye(p, dtype=normalized.dtype, device=normalized.device)
        identity = identity.expand(batch_size, -1, -1)
        inverse = torch.linalg.solve(profiled_scatter, identity)
        inverse_sample_inverse = inverse @ normalized @ inverse
        inverse_diagonal = torch.diagonal(inverse, dim1=-2, dim2=-1)
        isi_diagonal = torch.diagonal(inverse_sample_inverse, dim1=-2, dim2=-1)
        log_diagonal_score = 0.5 * diagonal * (inverse_diagonal - isi_diagonal)
        fisher_scale = (diagonal * inverse_diagonal / (2.0 ** 0.5)).clamp_min(self.eps)
        normalized_score = log_diagonal_score / fisher_scale

        root = diagonal.sqrt()
        residual = normalized - profiled_scatter
        whitened_residual = residual / (root.unsqueeze(-1) * root.unsqueeze(-2))
        residual_energy = self._residual_energy(whitened_residual)
        leverage = torch.diagonal(lowrank, dim1=-2, dim2=-1) / diagonal

        positions = torch.arange(p, device=normalized.device)
        signal_mask = positions.unsqueeze(0) >= (p - ranks).unsqueeze(1)
        bulk_mask = ~signal_mask
        excess = (eigenvalues - 1.0).clamp_min(0.0)
        signal_excess = (excess * signal_mask).sum(dim=-1) / ranks.to(normalized.dtype)
        signal_boundary = eigenvalues.gather(1, (p - ranks)[:, None]).squeeze(1)
        bulk_boundary = eigenvalues.gather(1, (p - ranks - 1).clamp_min(0)[:, None]).squeeze(1)
        gap = (signal_boundary - bulk_boundary).clamp_min(0.0)
        bulk_count = (p - ranks).clamp_min(1).to(normalized.dtype)
        bulk_energy = ((eigenvalues - 1.0).square() * bulk_mask).sum(dim=-1) / bulk_count
        global_features = torch.stack(
            (
                signal_excess.log1p(),
                gap.log1p(),
                bulk_energy.log1p(),
                torch.sqrt(normalized.new_tensor(float(p)) / sample_counts),
                ranks.to(normalized.dtype) / float(p),
            ),
            dim=-1,
        ).unsqueeze(1).expand(-1, p, -1)
        local_features = torch.stack(
            (
                diagonal.log(),
                (projected_diagonal / diagonal).clamp_min(self.eps).log(),
                normalized_score.asinh(),
                fisher_scale.log(),
                residual_energy,
                leverage.clamp_min(0.0).log1p(),
            ),
            dim=-1,
        )
        return torch.cat((local_features, global_features), dim=-1)

    def _features(
        self,
        normalized: torch.Tensor,
        diagonal: torch.Tensor,
        lowrank: torch.Tensor,
        projected_diagonal: torch.Tensor,
        eigenvalues: torch.Tensor,
        sample_counts: torch.Tensor,
        ranks: torch.Tensor,
    ) -> torch.Tensor:
        """Return 16 features: 6 local + 5 global + 5 post-projection local."""
        base_features = self._base_features(
            normalized, diagonal, lowrank, projected_diagonal, eigenvalues,
            sample_counts, ranks,
        )
        batch_size, p, _ = normalized.shape
        post_scatter = assemble_scatter(lowrank, projected_diagonal)
        identity = torch.eye(p, dtype=normalized.dtype, device=normalized.device)
        identity = identity.expand(batch_size, -1, -1)
        post_inverse = torch.linalg.solve(post_scatter, identity)
        post_inverse_sample_inverse = post_inverse @ normalized @ post_inverse
        post_inverse_diagonal = torch.diagonal(post_inverse, dim1=-2, dim2=-1)
        post_isi_diagonal = torch.diagonal(
            post_inverse_sample_inverse, dim1=-2, dim2=-1
        )
        post_log_score = 0.5 * projected_diagonal * (
            post_inverse_diagonal - post_isi_diagonal
        )
        post_fisher_scale = (
            projected_diagonal * post_inverse_diagonal / (2.0 ** 0.5)
        ).clamp_min(self.eps)
        post_normalized_score = post_log_score / post_fisher_scale
        post_root = projected_diagonal.sqrt()
        post_residual = normalized - post_scatter
        post_whitened_residual = post_residual / (
            post_root.unsqueeze(-1) * post_root.unsqueeze(-2)
        )
        post_residual_energy = self._residual_energy(post_whitened_residual)
        post_leverage = torch.diagonal(lowrank, dim1=-2, dim2=-1) / projected_diagonal
        post_features = torch.stack(
            (
                projected_diagonal.log(),
                post_normalized_score.asinh(),
                post_fisher_scale.log(),
                post_residual_energy,
                post_leverage.clamp_min(0.0).log1p(),
            ),
            dim=-1,
        )
        return torch.cat((base_features, post_features), dim=-1)

    def _raw_correction(
        self,
        features: torch.Tensor,
        normalized: torch.Tensor,
        lowrank_first: torch.Tensor,
        projected_diagonal: torch.Tensor,
    ) -> torch.Tensor:
        root = projected_diagonal.sqrt()
        scale = root.unsqueeze(-1) * root.unsqueeze(-2)
        post_scatter = assemble_scatter(lowrank_first, projected_diagonal)
        edges = torch.stack(
            (
                (normalized / scale).asinh(),
                (lowrank_first / scale).asinh(),
                ((normalized - post_scatter) / scale).asinh(),
            ),
            dim=-1,
        )
        return self.network(features, edges).squeeze(-1)

    def forward_components(
        self,
        sample_scatter: torch.Tensor,
        sample_counts: torch.Tensor,
        ranks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_inputs(sample_scatter, ranks)
        batch_size, _, _ = sample_scatter.shape
        if sample_counts.shape != (batch_size,) or bool((sample_counts <= 0).any()):
            raise ValueError("sample_counts must be a positive batch vector")
        scale = torch.diagonal(sample_scatter, dim1=-2, dim2=-1).mean(dim=-1)
        scale = scale.clamp_min(self.eps)
        normalized = sample_scatter / scale[:, None, None]
        diagonal = torch.diagonal(normalized, dim1=-2, dim2=-1).clamp_min(self.eps)
        lowrank_first, projected_diagonal, eigenvalues, _ = fad_projection(
            normalized, diagonal, ranks, eps=self.eps
        )
        features = self._features(
            normalized, diagonal, lowrank_first, projected_diagonal, eigenvalues,
            sample_counts, ranks,
        )
        raw_correction = self._raw_correction(
            features, normalized, lowrank_first, projected_diagonal
        )
        step_scale = functional.softplus(self.log_step_scale)
        predicted_diagonal = projected_diagonal * torch.exp(
            step_scale * torch.tanh(raw_correction)
        )
        predicted_diagonal = predicted_diagonal.clamp_min(self.eps)
        lowrank_final, _, _ = conditional_lowrank_with_spectrum(
            normalized, predicted_diagonal, ranks, eps=self.eps
        )
        normalized_estimate = assemble_scatter(lowrank_final, predicted_diagonal)
        return (
            normalized_estimate * scale[:, None, None],
            lowrank_final * scale[:, None, None],
            predicted_diagonal * scale[:, None],
            features,
            raw_correction,
        )

    def forward(
        self,
        sample_scatter: torch.Tensor,
        sample_counts: torch.Tensor,
        ranks: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_components(sample_scatter, sample_counts, ranks)[0]


class TraceShrinkageBackoff(nn.Module):
    """Learn a task-level, trace-preserving isotropic backoff coefficient."""

    def __init__(self, base: JointAdaptivePairwiseEstimator) -> None:
        super().__init__()
        self.base = base
        self.gate = nn.Sequential(
            nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 1)
        ).double()
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 4.0)

    def forward_components(
        self,
        sample_scatter: torch.Tensor,
        sample_counts: torch.Tensor,
        ranks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        estimate, lowrank, diagonal, features, _ = self.base.forward_components(
            sample_scatter, sample_counts, ranks
        )
        alpha = torch.sigmoid(self.gate(features[:, 0, 6:11]).squeeze(-1))
        trace_mean = torch.diagonal(estimate, dim1=-2, dim2=-1).mean(dim=-1)
        shrunk_lowrank = alpha[:, None, None] * lowrank
        shrunk_diagonal = (
            alpha[:, None] * diagonal
            + (1.0 - alpha[:, None]) * trace_mean[:, None]
        )
        shrunk_estimate = assemble_scatter(shrunk_lowrank, shrunk_diagonal)
        return shrunk_estimate, shrunk_lowrank, shrunk_diagonal, features, alpha

    def forward(
        self,
        sample_scatter: torch.Tensor,
        sample_counts: torch.Tensor,
        ranks: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_components(sample_scatter, sample_counts, ranks)[0]


def load_pairwise_checkpoint(checkpoint_path: str | Path) -> TraceShrinkageBackoff:
    """Load the principal RSE checkpoint into the self-contained architecture."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TraceShrinkageBackoff(JointAdaptivePairwiseEstimator().double()).double()
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
