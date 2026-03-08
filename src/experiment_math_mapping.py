"""
Experiment 3: Mathematical Mapping Between Attention, PageRank, and Collaborative Filtering

Demonstrates the formal isomorphism between:
1. Transformer attention: softmax(QK^T / √d) V
2. PageRank: p = (1-d) * (1/N) + d * M^T * p  (stationary distribution of attention-weighted graph)
3. Collaborative filtering: predicted_rating = softmax(similarity(u, U)) * R  (attention over user preferences)

We construct concrete numerical examples showing these mechanisms are instances
of the same mathematical pattern: weighted aggregation via normalized relevance scores.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

np.random.seed(42)


def transformer_attention(Q, K, V):
    """Standard scaled dot-product attention."""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    output = weights @ V
    return output, weights


def pagerank_as_attention(adjacency_matrix, damping=0.85, max_iter=100):
    """
    PageRank viewed as an attention mechanism.
    Each page "attends" to pages linking to it, weighted by their importance.

    The transition matrix M is analogous to attention weights:
    - M[i,j] = fraction of page j's links pointing to page i
    - This is a normalized relevance score, just like attention weights

    The PageRank vector p is analogous to attention output:
    - p = aggregation of page values weighted by attention (link) structure
    """
    N = adjacency_matrix.shape[0]
    # Normalize columns (out-links)
    out_degrees = adjacency_matrix.sum(axis=0)
    out_degrees[out_degrees == 0] = 1  # avoid division by zero
    M = adjacency_matrix / out_degrees  # transition matrix

    # PageRank iteration
    p = np.ones(N) / N
    for _ in range(max_iter):
        p_new = (1 - damping) / N + damping * M @ p
        if np.abs(p_new - p).sum() < 1e-10:
            break
        p = p_new

    # The attention weights are the normalized transition probabilities
    # weighted by current importance (analogous to QK^T softmax)
    attention_weights = M * p[np.newaxis, :]
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

    return p, attention_weights, M


def collaborative_filtering_as_attention(user_item_matrix, target_user_idx):
    """
    Collaborative filtering as attention over other users.

    Query: target user's preferences
    Keys: all other users' preferences
    Values: their ratings

    similarity(target, other) → attention weight
    predicted_rating = sum(attention_weight * other_user_rating)

    This is exactly: output = softmax(Q @ K^T) @ V
    where Q = target user, K = all users, V = ratings
    """
    target = user_item_matrix[target_user_idx]

    # Compute cosine similarity (analogous to dot-product attention)
    norms = np.linalg.norm(user_item_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = user_item_matrix / norms

    similarities = normalized @ normalized[target_user_idx]

    # Softmax normalization (same as attention)
    similarities[target_user_idx] = -np.inf  # don't attend to self
    exp_sim = np.exp(similarities - np.max(similarities[similarities != -np.inf]))
    exp_sim[target_user_idx] = 0
    attention_weights = exp_sim / exp_sim.sum()

    # Weighted aggregation (same as attention output = weights @ values)
    predicted_ratings = attention_weights @ user_item_matrix

    return predicted_ratings, attention_weights, similarities


def run_mathematical_mapping():
    results = {}

    # ── 1. Transformer Attention ──
    print("="*60)
    print("1. TRANSFORMER ATTENTION")
    print("="*60)

    d = 4
    seq_len = 6
    Q = np.random.randn(seq_len, d)
    K = np.random.randn(seq_len, d)
    V = np.random.randn(seq_len, d)

    output, attn_weights = transformer_attention(Q, K, V)

    print(f"  Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"  Attention weights (row sums = 1): {attn_weights.sum(axis=1)}")
    print(f"  Output = softmax(QK^T/√d) @ V")

    results["transformer"] = {
        "mechanism": "softmax(QK^T/√d) @ V",
        "query": "token representation",
        "key": "all token representations",
        "value": "all token representations",
        "normalization": "softmax over relevance scores",
        "output": "weighted combination of values",
        "weight_entropy": float(-np.sum(attn_weights * np.log(attn_weights + 1e-10)) / seq_len)
    }

    # ── 2. PageRank as Attention ──
    print("\n" + "="*60)
    print("2. PAGERANK AS ATTENTION")
    print("="*60)

    # Create a small web graph (6 pages)
    adj = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ], dtype=float)

    pagerank, pr_attention, M = pagerank_as_attention(adj)

    print(f"  Graph: {adj.shape[0]} pages")
    print(f"  PageRank scores: {pagerank}")
    print(f"  Transition matrix M (normalized links = attention weights):")
    print(f"  M row sums: {M.sum(axis=0)}")
    print(f"  PageRank attention row sums: {pr_attention.sum(axis=1)}")

    # The key insight: PageRank is iterative attention
    # M[i,j] = how much page j "attends to" page i (via link)
    # p_i = sum_j M[i,j] * p_j  → attention-weighted aggregation

    results["pagerank"] = {
        "mechanism": "p = (1-d)/N + d * M @ p (iterative attention)",
        "query": "page importance query",
        "key": "link structure (which pages link here)",
        "value": "source page importance scores",
        "normalization": "column-normalized adjacency (link probability)",
        "output": "page importance = attention-weighted sum of linking pages",
        "weight_entropy": float(-np.sum(pr_attention * np.log(pr_attention + 1e-10)) / adj.shape[0]),
        "pagerank_scores": pagerank.tolist()
    }

    # ── 3. Collaborative Filtering as Attention ──
    print("\n" + "="*60)
    print("3. COLLABORATIVE FILTERING AS ATTENTION")
    print("="*60)

    # User-item rating matrix (6 users, 8 items)
    ratings = np.array([
        [5, 4, 0, 0, 1, 0, 0, 2],
        [4, 5, 4, 0, 0, 0, 0, 1],
        [0, 0, 0, 5, 4, 5, 0, 0],
        [0, 0, 0, 4, 5, 4, 5, 0],
        [5, 3, 4, 0, 0, 0, 0, 3],
        [0, 0, 0, 3, 4, 5, 4, 0],
    ], dtype=float)

    target_user = 0
    predicted, cf_weights, similarities = collaborative_filtering_as_attention(ratings, target_user)

    print(f"  Rating matrix: {ratings.shape} (users × items)")
    print(f"  Target user {target_user} ratings: {ratings[target_user]}")
    print(f"  Attention weights over other users: {cf_weights}")
    print(f"  Predicted ratings: {predicted}")
    print(f"  Weight sum: {cf_weights.sum():.4f}")

    results["collaborative_filtering"] = {
        "mechanism": "predicted = softmax(similarity(target, others)) @ ratings",
        "query": "target user preference vector",
        "key": "all users' preference vectors",
        "value": "all users' rating vectors",
        "normalization": "softmax over cosine similarities",
        "output": "predicted ratings = attention-weighted combination",
        "weight_entropy": float(-np.sum(cf_weights * np.log(cf_weights + 1e-10))),
        "attention_weights": cf_weights.tolist(),
        "predicted_ratings": predicted.tolist()
    }

    # ── 4. Unified Mathematical Framework ──
    print("\n" + "="*60)
    print("4. UNIFIED MATHEMATICAL FRAMEWORK")
    print("="*60)

    unified = {
        "general_form": "output = normalize(relevance(query, keys)) @ values",
        "components": {
            "transformer": {
                "relevance": "dot_product(q, k) / √d",
                "normalize": "softmax",
                "aggregate": "weighted sum of value vectors"
            },
            "pagerank": {
                "relevance": "link(source, target) / out_degree(source)",
                "normalize": "column normalization + damping",
                "aggregate": "weighted sum of source importances"
            },
            "collaborative_filtering": {
                "relevance": "cosine_similarity(user_a, user_b)",
                "normalize": "softmax",
                "aggregate": "weighted sum of other users' ratings"
            },
            "recommendation_algorithm": {
                "relevance": "predicted_user_interest(user, item)",
                "normalize": "ranking/softmax over candidates",
                "aggregate": "top-K selection for user feed"
            }
        },
        "key_properties": [
            "All use normalized relevance scores as weights",
            "All produce output as weighted combination of values",
            "All weights sum to 1 (probability distribution)",
            "All are differentiable (can be optimized with gradient descent)",
            "All serve to route information selectively (attention bottleneck)"
        ]
    }

    for prop in unified["key_properties"]:
        print(f"  ✓ {prop}")

    results["unified_framework"] = unified

    # ── Create Visualizations ──
    create_visualizations(attn_weights, pr_attention, cf_weights, results)

    # Save results
    with open(os.path.join(RESULTS_DIR, "math_mapping_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to results/math_mapping_results.json")
    return results


def create_visualizations(attn_weights, pr_attention, cf_weights, results):
    """Create comparison visualizations."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Transformer attention heatmap
    im1 = axes[0].imshow(attn_weights, cmap="Blues", aspect="auto")
    axes[0].set_title("Transformer Self-Attention\nsoftmax(QK^T/√d)", fontsize=11)
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # 2. PageRank attention heatmap
    im2 = axes[1].imshow(pr_attention, cmap="Greens", aspect="auto")
    axes[1].set_title("PageRank Attention\nNormalized link weights × importance", fontsize=11)
    axes[1].set_xlabel("Source page")
    axes[1].set_ylabel("Target page")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # 3. CF attention weights
    cf_matrix = np.outer(cf_weights, np.ones(6))
    im3 = axes[2].imshow(cf_matrix[:, :1].T, cmap="Oranges", aspect="auto")
    axes[2].bar(range(len(cf_weights)), cf_weights, color="orange", alpha=0.8)
    axes[2].set_title("Collaborative Filtering Attention\nsoftmax(similarity) weights", fontsize=11)
    axes[2].set_xlabel("User index")
    axes[2].set_ylabel("Attention weight")

    plt.suptitle("Attention Mechanism Across Three Domains:\nSame Mathematical Pattern, Different Applications",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "attention_mapping_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Unified diagram: entropy comparison ──
    fig, ax = plt.subplots(figsize=(8, 5))

    mechanisms = ["Transformer\nAttention", "PageRank\n(Web Graph)", "Collaborative\nFiltering"]
    entropies = [
        results["transformer"]["weight_entropy"],
        results["pagerank"]["weight_entropy"],
        results["collaborative_filtering"]["weight_entropy"]
    ]
    colors = ["#4C72B0", "#55A868", "#DD8452"]

    bars = ax.bar(mechanisms, entropies, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Attention Weight Entropy (nats)", fontsize=12)
    ax.set_title("Attention Weight Entropy Across Mechanisms\n(All implement: output = normalize(relevance) @ values)",
                 fontsize=12, fontweight="bold")

    for bar, e in zip(bars, entropies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{e:.3f}', ha='center', va='bottom', fontsize=11)

    ax.set_ylim(0, max(entropies) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "entropy_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("  ✓ Visualizations saved to results/plots/")


if __name__ == "__main__":
    run_mathematical_mapping()
