"""
Generate all analysis visualizations from experiment results.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_cross_domain_comparison():
    """Bar chart comparing attention vs non-attention baselines across domains."""
    with open(os.path.join(RESULTS_DIR, "cross_domain_results.json")) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # IMDB
    imdb = results["imdb"]
    models = ["Transformer", "BiLSTM", "Bag-of-Words"]
    keys = ["transformer", "lstm", "bow"]
    means = [imdb[k]["mean"] for k in keys]
    stds = [imdb[k]["std"] for k in keys]
    colors = ["#4C72B0", "#C44E52", "#8C8C8C"]

    bars = axes[0].bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                       edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("IMDB Sentiment Classification\n(Text Domain)", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 1.0)
    for bar, m in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                     f'{m:.3f}', ha='center', va='bottom', fontsize=10)

    # CIFAR-10
    cifar = results["cifar10"]
    models = ["Transformer (ViT)", "CNN"]
    keys = ["transformer", "cnn"]
    means = [cifar[k]["mean"] for k in keys]
    stds = [cifar[k]["std"] for k in keys]
    colors = ["#4C72B0", "#C44E52"]

    bars = axes[1].bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                       edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("CIFAR-10 Image Classification\n(Vision Domain)", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 1.0)
    for bar, m in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                     f'{m:.3f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle("Attention-Based vs Non-Attention Models Across Domains",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cross_domain_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Cross-domain comparison plot saved")


def plot_attention_entropy_distributions():
    """Compare attention entropy distributions across domains."""
    with open(os.path.join(RESULTS_DIR, "attention_distributions.json")) as f:
        dists = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))

    for domain, entropies in dists.items():
        if entropies:
            label = {"imdb": "Text (IMDB)", "cifar10": "Vision (CIFAR-10)"}[domain]
            color = {"imdb": "#4C72B0", "cifar10": "#55A868"}[domain]
            ax.hist(entropies, bins=50, alpha=0.5, label=label, color=color, density=True)

    ax.set_xlabel("Attention Entropy (nats)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Attention Entropy Distribution Across Modalities\n"
                 "(Similar distributions suggest universal attention behavior)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "attention_entropy_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Attention entropy distribution plot saved")


def plot_unified_framework():
    """Create a summary visualization of the unified attention framework."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    # Title
    ax.text(0.5, 0.95, "Unified Attention Framework",
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.90, "output = normalize( relevance(query, keys) ) @ values",
            ha="center", va="top", fontsize=14, family="monospace",
            style="italic", color="#333333", transform=ax.transAxes)

    # Table data
    domains = [
        ("Transformer\n(Generative AI)", "#4C72B0",
         "Token repr.", "All tokens", "Token values",
         "softmax(QK^T/√d)", "Contextual repr."),
        ("PageRank\n(Web/Internet)", "#55A868",
         "Page query", "Link structure", "Page importance",
         "Norm. links + damping", "Page rank score"),
        ("Rec. Systems\n(Internet)", "#DD8452",
         "User prefs.", "All users' prefs.", "User ratings",
         "softmax(cosine sim)", "Predicted ratings"),
        ("Search Engines\n(Internet)", "#937860",
         "Search query", "Doc. features", "Doc. content",
         "BM25/learned", "Ranked results"),
    ]

    headers = ["Domain", "Query (Q)", "Keys (K)", "Values (V)", "Normalization", "Output"]
    col_x = [0.05, 0.20, 0.35, 0.50, 0.67, 0.85]

    y = 0.80
    for i, h in enumerate(headers):
        ax.text(col_x[i], y, h, ha="center", va="center", fontsize=11,
                fontweight="bold", transform=ax.transAxes)

    for j, (domain, color, q, k, v, norm, out) in enumerate(domains):
        y = 0.68 - j * 0.14
        vals = [domain, q, k, v, norm, out]
        for i, val in enumerate(vals):
            fc = color if i == 0 else "white"
            tc = "white" if i == 0 else "black"
            bbox = dict(boxstyle="round,pad=0.3", facecolor=fc, alpha=0.8, edgecolor=color)
            ax.text(col_x[i], y, val, ha="center", va="center", fontsize=9,
                    transform=ax.transAxes, bbox=bbox, color=tc)

    # Bottom note
    ax.text(0.5, 0.12, "Key Insight: All mechanisms implement the same pattern —\n"
            "selective information routing through normalized relevance weighting.\n"
            "This is attention: filtering information through a bottleneck of limited capacity.",
            ha="center", va="center", fontsize=11, style="italic",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc"))

    plt.savefig(os.path.join(PLOTS_DIR, "unified_framework.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Unified framework diagram saved")


def generate_all_plots():
    plot_cross_domain_comparison()
    plot_attention_entropy_distributions()
    plot_unified_framework()
    print("\n✓ All visualizations generated in results/plots/")


if __name__ == "__main__":
    generate_all_plots()
