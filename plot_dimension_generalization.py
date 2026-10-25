"""Plot the SCM-versus-PANDA dimension-generalization result saved by evaluation."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PACKAGE_ROOT / "results" / "dimension_generalization_scm_panda.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results" / "dimension_generalization_scm_panda.png"


def configure_style() -> None:
    plt.rcParams.update({
        "axes.edgecolor": "#4D4D4D", "axes.labelcolor": "#202020", "axes.labelsize": 16,
        "axes.linewidth": 0.8, "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 14, "legend.fontsize": 10, "mathtext.fontset": "stix",
        "savefig.dpi": 300, "savefig.facecolor": "white", "xtick.color": "#202020",
        "xtick.direction": "out", "xtick.labelsize": 12, "ytick.color": "#202020",
        "ytick.direction": "out", "ytick.labelsize": 12,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = json.loads(arguments.input.read_text())
    dimensions = payload["protocol"]["test_dimensions"]
    train_dimensions = payload["protocol"]["train_dimensions"]
    results = payload["results"]
    configure_style()
    figure, axis = plt.subplots(figsize=(6.2, 4.15))
    train_end = max(train_dimensions) + 0.5
    unseen_start = min(p for p in dimensions if p not in train_dimensions) - 0.5
    axis.axvspan(min(dimensions) - 0.5, train_end, ymin=0.895, ymax=1.0, color="#DCEAF7", zorder=0)
    axis.axvspan(unseen_start, max(dimensions) + 0.5, ymin=0.895, ymax=1.0, color="#FBE7C6", zorder=0)
    axis.axvline((train_end + unseen_start) / 2, color="#666666", linestyle=(0, (5, 3)), linewidth=1.1)
    styles = {
        "SCM": {"color": "#666666", "linestyle": "-", "marker": "P", "linewidth": 1.65},
        "PANDA": {"color": "#CC3311", "linestyle": "-", "marker": "o", "linewidth": 2.1},
    }
    labels = {"SCM": "SCM", "PANDA": "PANDA (prop.)"}
    for method in ("SCM", "PANDA"):
        means = [results[str(p)][method]["mean"] for p in dimensions]
        axis.plot(dimensions, means, label=labels[method], markersize=6.0,
                  markeredgecolor="white", markeredgewidth=0.55, solid_capstyle="round", **styles[method])
    axis.set_facecolor("white")
    axis.set_axisbelow(True)
    axis.grid(True, which="major", color="#D9D9D9", linestyle=(0, (1.2, 2.2)), linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_color("#4D4D4D")
        spine.set_linewidth(0.75)
    axis.set_xlim(min(dimensions) - 0.5, max(dimensions) + 0.5)
    axis.set_xticks(dimensions)
    axis.set_xlabel(r"$p$")
    axis.set_ylabel("Mean relative squared error")
    axis.margins(y=0.10)
    axis.text(0.12, 0.947, "Training", transform=axis.transAxes, ha="center", va="center", fontsize=10, color="#243746")
    axis.text(0.51, 0.947, "Unseen", transform=axis.transAxes, ha="center", va="center", fontsize=10, color="#5C3D1E")
    legend = axis.legend(loc="upper right", bbox_to_anchor=(0.985, 0.875), frameon=True, fancybox=False,
                         handlelength=2.25, borderpad=0.5)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#BFBFBF")
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_alpha(1.0)
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.97)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output.with_suffix(".png"), bbox_inches="tight")
    plt.close(figure)
    print(f"saved={arguments.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
