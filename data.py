"""Synthetic low-rank-plus-diagonal covariance tasks used in the PANDA example.

The evaluation generator is the exact extreme-diagonal-heterogeneity generator
used for the published dimension-generalization panel.  All returned tensors
use float64 so that CPU results are deterministic for the supplied seeds.
"""

import torch


DTYPE = torch.float64
TEST_DIMENSIONS = (6, 8, 10, 12, 14, 16, 18, 20)
ASPECT_RATIOS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SUPPORTED_RANKS = (1, 2, 3)
SCENARIO = {
    "name": "extreme_diagonal_heterogeneity",
    "snr_range_db": (-30.0, 10.0),
    "spread_range": (1.6, 2.4),
    "sample_family": "gaussian_iid",
}


def relative_squared_frobenius_error(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return ||estimate-target||_F^2 / ||target||_F^2 for every task."""
    numerator = (estimate - target).square().sum(dim=(-2, -1))
    denominator = target.square().sum(dim=(-2, -1)).clamp_min(1e-12)
    return numerator / denominator


def generate_tasks(
    size: int,
    p: int,
    sample_count: int,
    seed: int,
    fixed_rank: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate sample scatters and their true low-rank-plus-diagonal covariance.

    The diagonal varies across log-normal, Gamma, and two-level distributions;
    loadings vary across Gaussian, Rademacher, and sparse distributions.  This
    is intentionally a task generator, not a fixed train/test data file.
    """
    if p < 2 or sample_count < 2:
        raise ValueError("p and sample_count must both be at least two")
    if fixed_rank is not None and fixed_rank not in SUPPORTED_RANKS:
        raise ValueError(f"fixed_rank must be one of {SUPPORTED_RANKS}")
    if fixed_rank is not None and fixed_rank >= p:
        raise ValueError("fixed_rank must be smaller than p")

    generator = torch.Generator().manual_seed(seed)
    index = torch.arange(size)
    diagonal_id = index.remainder(3)
    loading_id = (index // 3).remainder(3)
    rank_id = (index // 9).remainder(len(SUPPORTED_RANKS))
    ranks = torch.tensor(SUPPORTED_RANKS, dtype=torch.long)[rank_id]
    if fixed_rank is not None:
        ranks = torch.full((size,), fixed_rank, dtype=torch.long)

    spread = torch.empty(size, dtype=DTYPE).uniform_(
        *SCENARIO["spread_range"], generator=generator
    )
    lognormal = torch.exp(spread[:, None] * torch.randn(size, p, generator=generator, dtype=DTYPE))
    gamma = torch._standard_gamma(
        torch.full((size, p), 2.0, dtype=DTYPE), generator=generator
    ).pow(0.5 + spread[:, None] / 2.0)
    levels = 2 * torch.randint(0, 2, (size, p), generator=generator).to(DTYPE) - 1
    two_level = torch.exp(spread[:, None] * levels)
    diagonal = torch.where(
        (diagonal_id == 0)[:, None], lognormal,
        torch.where((diagonal_id == 1)[:, None], gamma, two_level),
    ).clamp_min(1e-8)
    diagonal = diagonal / diagonal.mean(dim=-1, keepdim=True)

    max_rank = max(SUPPORTED_RANKS)
    gaussian = torch.randn(size, p, max_rank, generator=generator, dtype=DTYPE)
    rademacher = 2 * torch.randint(0, 2, (size, p, max_rank), generator=generator).to(DTYPE) - 1
    sparse = torch.randn(size, p, max_rank, generator=generator, dtype=DTYPE)
    sparse_mask = torch.rand(size, p, max_rank, generator=generator, dtype=DTYPE) < 0.45
    sparse_mask[:, 0, :] |= ~sparse_mask.any(dim=1)
    sparse = sparse * sparse_mask
    loading = torch.where(
        (loading_id == 0)[:, None, None], gaussian,
        torch.where((loading_id == 1)[:, None, None], rademacher, sparse),
    )
    active = torch.arange(max_rank)[None, None, :] < ranks[:, None, None]
    loading = loading * active.to(DTYPE)
    snr_db = torch.empty(size, dtype=DTYPE).uniform_(*SCENARIO["snr_range_db"], generator=generator)
    loading_scale = torch.sqrt(
        diagonal.sum(dim=-1) * torch.pow(10.0, snr_db / 10.0)
        / loading.square().sum(dim=(-2, -1)).clamp_min(1e-12)
    )
    loading = loading * loading_scale[:, None, None]
    target = loading @ loading.transpose(-2, -1) + torch.diag_embed(diagonal)
    observations = torch.randn(size, sample_count, p, generator=generator, dtype=DTYPE)
    observations = observations @ torch.linalg.cholesky(target).transpose(-2, -1)
    sample_scatter = observations.transpose(-2, -1) @ observations / float(sample_count)
    counts = torch.full((size,), sample_count, dtype=DTYPE)
    return sample_scatter, target, ranks, counts


def task_sets(
    dimensions: tuple[int, ...], tasks_per_stratum: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate the p-stratified task sets used by the fixed training schedule."""
    datasets = []
    for p in dimensions:
        parts = [[], [], [], []]
        for ratio_index, ratio in enumerate(ASPECT_RATIOS):
            sample_count = int(p * ratio)
            generated = generate_tasks(
                tasks_per_stratum, p, sample_count,
                seed + 100000 * p + 1000 * ratio_index,
            )
            for destination, value in zip(parts, generated):
                destination.append(value)
        datasets.append(tuple(torch.cat(values) for values in parts))
    return datasets
