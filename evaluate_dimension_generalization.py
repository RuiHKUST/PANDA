"""Evaluate SCM and the supplied PANDA checkpoint on the dimension sweep."""

import argparse
import json
from pathlib import Path

import torch

from data import TEST_DIMENSIONS, generate_tasks, relative_squared_frobenius_error
from panda_model import load_pairwise_checkpoint


PACKAGE_ROOT = Path(__file__).resolve().parent
TRAIN_DIMENSIONS = (6, 8)
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "panda_dimension_generalization.pt"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results" / "dimension_generalization_scm_panda.json"


def summary(values: list[torch.Tensor]) -> dict[str, float]:
    merged = torch.cat(values)
    return {
        "mean": merged.mean().item(),
        "standard_error": (merged.std(unbiased=True) / merged.numel() ** 0.5).item(),
        "tasks": int(merged.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks-per-seed", type=int, default=12)
    parser.add_argument("--first-seed", type=int, default=20262001)
    parser.add_argument("--num-seeds", type=int, default=100)
    arguments = parser.parse_args()
    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(arguments.checkpoint)
    torch.set_num_threads(1)
    model = load_pairwise_checkpoint(arguments.checkpoint)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for p in TEST_DIMENSIONS:
        errors = {"SCM": [], "PANDA": []}
        for seed_index in range(arguments.num_seeds):
            seed = arguments.first_seed + seed_index + 1000 * p
            scatter, target, ranks, counts = generate_tasks(
                arguments.tasks_per_seed, p, sample_count=p, seed=seed, fixed_rank=1
            )
            with torch.inference_mode():
                panda = model(scatter, counts, ranks)
            errors["SCM"].append(relative_squared_frobenius_error(scatter, target))
            errors["PANDA"].append(relative_squared_frobenius_error(panda, target))
        results[str(p)] = {method: summary(values) for method, values in errors.items()}
        print(f"p={p:2d} SCM={results[str(p)]['SCM']['mean']:.6f} PANDA={results[str(p)]['PANDA']['mean']:.6f}")
    payload = {
        "protocol": {
            "scenario": "extreme_diagonal_heterogeneity",
            "train_dimensions": list(TRAIN_DIMENSIONS),
            "test_dimensions": list(TEST_DIMENSIONS),
            "fixed_n_over_p": 1,
            "fixed_rank": 1,
            "first_seed": arguments.first_seed,
            "num_seeds": arguments.num_seeds,
            "tasks_per_seed": arguments.tasks_per_seed,
            "checkpoint": "checkpoints/panda_dimension_generalization.pt",
            "metric": "relative squared Frobenius error",
        },
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved={arguments.output}")


if __name__ == "__main__":
    main()
