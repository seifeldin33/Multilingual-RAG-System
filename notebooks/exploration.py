"""
Data exploration and visualization script.

Generates statistics and plots about the Natural Questions dataset,
saving results to the output/ directory.
"""

import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter

from data.processing import load_and_clean, classify_question_type, classify_domain

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def explore_dataset(sample_size: int = 0):
    """
    Run full EDA on the dataset: compute stats and generate plots.

    Parameters
    ----------
    sample_size : int
        Number of rows to use (0 = full dataset).
    """
    ensure_output_dir()

    # ── Load & Clean ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Multilingual RAG — Dataset Exploration")
    print("=" * 60)

    df = load_and_clean(sample_size=sample_size)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # ── Basic Stats ───────────────────────────────────────────────────────
    print("\n── Basic Statistics ──")
    print(f"  Total rows:              {len(df):,}")
    print(f"  Rows with long answer:   {df['long_answer'].astype(bool).sum():,}")
    print(f"  Rows with short answer:  {df['short_answer'].astype(bool).sum():,}")

    df["question_len"] = df["question"].str.len()
    df["long_answer_len"] = df["long_answer"].str.len()
    df["short_answer_len"] = df["short_answer"].str.len()

    print(f"\n  Avg question length:     {df['question_len'].mean():.0f} chars")
    print(f"  Avg long answer length:  {df['long_answer_len'].mean():.0f} chars")
    print(f"  Avg short answer length: {df['short_answer_len'].mean():.0f} chars")

    # ── Question Type Distribution ────────────────────────────────────────
    print("\n── Question Type Distribution ──")
    df["question_type"] = df["question"].apply(classify_question_type)
    type_counts = df["question_type"].value_counts()
    for qt, count in type_counts.items():
        pct = count / len(df) * 100
        print(f"  {qt:<15} {count:>6,}  ({pct:.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(type_counts))
    type_counts.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Question Type Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "question_types.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved question_types.png")

    # ── Domain Distribution ───────────────────────────────────────────────
    print("\n── Domain Distribution ──")
    df["domain"] = df.apply(
        lambda r: classify_domain(f"{r['question']} {r['long_answer']}"), axis=1
    )
    domain_counts = df["domain"].value_counts()
    for domain, count in domain_counts.items():
        pct = count / len(df) * 100
        print(f"  {domain:<15} {count:>6,}  ({pct:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 8))
    # Filter out very small slices
    threshold = 0.02 * len(df)
    major = domain_counts[domain_counts >= threshold]
    other = domain_counts[domain_counts < threshold].sum()
    if other > 0:
        major["other (combined)"] = other
    colors = sns.color_palette("Set2", len(major))
    major.plot(kind="pie", ax=ax, autopct="%1.1f%%", colors=colors, startangle=140)
    ax.set_title("Domain Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "domain_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved domain_distribution.png")

    # ── Answer Length Distribution ─────────────────────────────────────────
    print("\n── Answer Length Distribution ──")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    long_lens = df["long_answer_len"][df["long_answer_len"] > 0]
    axes[0].hist(long_lens, bins=50, color="#4C72B0", alpha=0.8, edgecolor="white")
    axes[0].set_title("Long Answer Length", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(long_lens.median(), color="red", linestyle="--", label=f"Median: {long_lens.median():.0f}")
    axes[0].legend()

    short_lens = df["short_answer_len"][df["short_answer_len"] > 0]
    axes[1].hist(short_lens, bins=50, color="#55A868", alpha=0.8, edgecolor="white")
    axes[1].set_title("Short Answer Length", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Characters")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(short_lens.median(), color="red", linestyle="--", label=f"Median: {short_lens.median():.0f}")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "answer_lengths.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved answer_lengths.png")

    # ── Sample Questions Per Category ─────────────────────────────────────
    print("\n── Sample Questions ──")
    for qt in type_counts.index[:5]:
        sample = df[df["question_type"] == qt]["question"].sample(min(3, len(df[df["question_type"] == qt])), random_state=42)
        print(f"\n  [{qt}]")
        for q in sample:
            print(f"    • {q[:100]}")

    # ── Summary Report ────────────────────────────────────────────────────
    report = {
        "total_rows": len(df),
        "rows_with_long_answer": int(df["long_answer"].astype(bool).sum()),
        "rows_with_short_answer": int(df["short_answer"].astype(bool).sum()),
        "avg_question_length": float(df["question_len"].mean()),
        "avg_long_answer_length": float(df["long_answer_len"].mean()),
        "avg_short_answer_length": float(df["short_answer_len"].mean()),
        "question_types": type_counts.to_dict(),
        "domains": domain_counts.to_dict(),
    }

    import json
    with open(OUTPUT_DIR / "exploration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  → Saved exploration_report.json")
    print("\n" + "=" * 60)
    print("  Exploration complete! Check the output/ directory.")
    print("=" * 60)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    explore_dataset()
