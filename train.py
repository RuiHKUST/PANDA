"""Train PANDA from scratch on the public extreme-heterogeneity task family."""

import argparse
from pathlib import Path

import torch

from data import relative_squared_frobenius_error, task_sets
from panda_model import JointAdaptivePairwiseEstimator, TraceShrinkageBackoff


PACKAGE_ROOT = Path(__file__).resolve().parent


def batches(datasets, batch_size: int, generator: torch.Generator):
    queues = [torch.randperm(data[2].numel(), generator=generator).split(batch_size) for data in datasets]
    for round_index in range(max(len(queue) for queue in queues)):
        for position in torch.randperm(len(datasets), generator=generator).tolist():
            if round_index < len(queues[position]):
                yield datasets[position], queues[position][round_index]


def mean_error(model, datasets) -> torch.Tensor:
    model.eval()
    values = []
    with torch.inference_mode():
        for scatter, target, ranks, counts in datasets:
            values.append(relative_squared_frobenius_error(model(scatter, counts, ranks), target))
    return torch.cat(values).mean()


def optimize(
    model, training, epochs: int, learning_rate: float, batch_size: int, seed: int,
    stage: str,
) -> None:
    """Run one fixed-length optimization stage using training tasks only."""
    scale = mean_error(model, training).clamp_min(1e-12)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    generator = torch.Generator().manual_seed(seed + 1000)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for data, indices in batches(training, batch_size, generator):
            scatter, target, ranks, counts = data
            loss = relative_squared_frobenius_error(
                model(scatter[indices], counts[indices], ranks[indices]), target[indices]
            ).mean() / scale
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            mean_loss = torch.stack(losses).mean().item()
            print(f"stage={stage} epoch={epoch:03d} normalized_training_loss={mean_loss:.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small smoke-test configuration")
    parser.add_argument("--tasks-per-stratum", type=int, default=None)
    parser.add_argument("--base-epochs", type=int, default=None)
    parser.add_argument("--backoff-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261850)
    parser.add_argument(
        "--output", type=Path,
        default=PACKAGE_ROOT / "checkpoints" / "panda_dimension_generalization.pt",
    )
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(arguments.seed)
    quick = arguments.quick
    tasks = arguments.tasks_per_stratum or (64 if quick else 1000)
    base_epochs = arguments.base_epochs or (4 if quick else 60)
    backoff_epochs = arguments.backoff_epochs or (3 if quick else 20)
    training = task_sets((6, 8), tasks, arguments.seed)

    base = JointAdaptivePairwiseEstimator(hidden=48).double()
    optimize(
        base, training, base_epochs, learning_rate=2e-3, batch_size=128,
        seed=arguments.seed, stage="pairwise_warmup",
    )
    model = TraceShrinkageBackoff(base).double()
    optimize(
        model, training, backoff_epochs, learning_rate=7e-4, batch_size=128,
        seed=arguments.seed + 1, stage="panda_backoff",
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "protocol": {
            "architecture": "two-expert Pairwise message passing plus trace-preserving backoff",
            "objective": "relative squared Frobenius error",
            "train_dimensions": [6, 8],
            "tasks_per_stratum": tasks,
            "base_epochs": base_epochs,
            "backoff_epochs": backoff_epochs,
            "seed": arguments.seed,
            "selection": "none; final epoch of each fixed-length training stage",
        },
    }, arguments.output)
    print(f"saved={arguments.output}")


if __name__ == "__main__":
    main()
