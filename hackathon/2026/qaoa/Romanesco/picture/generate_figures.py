from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-moo-paper")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


RESULTS = PACKAGE_ROOT / "results"
PICTURE = ROOT
PICTURE.mkdir(exist_ok=True)


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
    }
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def total_score(data: dict) -> float:
    return float(data["score_k5"]) + float(data["score_large_bonus"])


def annotate_points(ax, xs, ys, fmt="{:.1f}", dy=3.0):
    for x, y in zip(xs, ys):
        ax.annotate(
            fmt.format(y),
            (x, y),
            textcoords="offset points",
            xytext=(0, dy),
            ha="center",
            fontsize=7,
        )


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PICTURE / name, bbox_inches="tight")
    plt.close(fig)


def plot_round_sweep(
    labels,
    scores,
    runtimes,
    title: str,
    output_name: str,
    highlight_index: int | None = None,
    highlight_label: str | None = None,
) -> None:
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax2 = ax.twinx()
    bars = ax2.bar(xs, runtimes, color="#C9D6DF", width=0.56, label="time / s")
    (line,) = ax.plot(xs, scores, color="#1F77B4", marker="o", linewidth=2.1, label="score_k5")

    annotate_points(ax, xs, scores, "{:.1f}", dy=5.0)
    ax.set_title(title)
    ax.set_xlabel("round")
    ax.set_ylabel("score_k5")
    ax2.set_ylabel("time / s")
    ax.set_xticks(xs, labels)
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    if highlight_index is not None:
        ax.axvline(highlight_index, color="#54A24B", linestyle="--", linewidth=1.2, alpha=0.8)
        if highlight_label:
            ax.text(
                highlight_index,
                max(scores),
                highlight_label,
                va="bottom",
                ha="center",
                fontsize=7,
                color="#3D7F38",
            )

    ax.legend([line, bars], ["score_k5", "time / s"], loc="upper left")
    save(fig, output_name)


def figure_round_sweeps() -> None:
    coord_specs = [
        ("1", "answer_multiround_score_1round.json"),
        ("2", "answer_multiround_score_2round.json"),
        ("3", "answer_multiround_score_3round.json"),
        ("4", "answer_multiround_score_4round.json"),
        ("5", "answer_multiround_score_5round.json"),
        ("6", "answer_multiround_score_6round.json"),
    ]
    coord_data = [
        read_json(RESULTS / "answer_multiround_score" / filename)
        for _, filename in coord_specs
    ]
    coord_labels = [label for label, _ in coord_specs]
    coord_x = list(range(len(coord_specs)))
    coord_scores = [d["score_k5"] for d in coord_data]
    coord_times = [d["elapsed"] for d in coord_data]

    orig_rounds = list(range(1, 7))
    orig_data = [
        read_json(RESULTS / "original_answer_round_sweep" / f"round_{r}.json")
        for r in orig_rounds
    ]
    orig_scores = [d["score_k5"] for d in orig_data]
    orig_times = [d["elapsed"] for d in orig_data]

    plot_round_sweep(
        coord_labels,
        coord_scores,
        coord_times,
        "Round sweep - coordinated feedback",
        "coord_round_sweep.png",
        highlight_index=3,
    )
    plot_round_sweep(
        [str(r) for r in orig_rounds],
        orig_scores,
        orig_times,
        "Round sweep - answer",
        "answer_round_sweep.png",
        highlight_index=4,
    )


def figure_main2_runtime() -> None:
    data = read_json(RESULTS / "main2_compare_latest_score.json")
    order = ["baseline_main2", "vectorized_pipeline", "numba_refinement"]
    names = ["answer / baseline", "vectorized", "numba"]
    runtimes = [data["results"][key]["avg_elapsed_s"] for key in order]
    bonus = [data["results"][key]["score_large_bonus"] for key in order]
    colors = ["#BAB0AC", "#4C78A8", "#54A24B", "#F58518"]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(names, runtimes, color=colors, width=0.6)
    ax.set_ylabel("average time / s")
    ax.set_title("main2 large10 time comparison")
    ax.set_ylim(0, max(runtimes) * 1.22)
    for bar, value, b in zip(bars, runtimes, bonus):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.45,
            f"{value:.2f}s\n+{b:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save(fig, "main2_runtime_comparison.png")


if __name__ == "__main__":
    figure_round_sweeps()
    figure_main2_runtime()
    print(f"Wrote figures to {PICTURE}")
