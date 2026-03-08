"""
Experiment 1 & 2: Cross-Domain Attention Universality
Trains small transformers and non-attention baselines on text (IMDB) and vision (CIFAR-10),
then extracts attention patterns for cross-domain analysis.
Uses proper train/test splits with enough data for meaningful comparison.
"""

import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
SEED = 42
NUM_SEEDS = 3

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────
# Data Loading - Use HuggingFace datasets for proper splits
# ─────────────────────────────────────────────────────

def load_imdb_data(max_len=128, vocab_size=5000, n_train=5000, n_test=2000):
    """Load IMDB from HuggingFace with proper train/test split."""
    from datasets import load_dataset
    print("  Loading IMDB dataset from HuggingFace...")
    ds = load_dataset("stanfordnlp/imdb")

    # Sample subsets
    train_ds = ds["train"].shuffle(seed=SEED).select(range(n_train))
    test_ds = ds["test"].shuffle(seed=SEED).select(range(n_test))

    train_texts = train_ds["text"]
    train_labels = train_ds["label"]
    test_texts = test_ds["text"]
    test_labels = test_ds["label"]

    # Build vocab from training data only
    all_words = []
    for t in train_texts:
        all_words.extend(t.lower().split()[:max_len])
    word_counts = Counter(all_words)
    vocab = {w: i+2 for i, (w, _) in enumerate(word_counts.most_common(vocab_size-2))}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1

    def encode(text):
        tokens = [vocab.get(w, 1) for w in text.lower().split()[:max_len]]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        return tokens

    X_train = torch.tensor([encode(t) for t in train_texts], dtype=torch.long)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_test = torch.tensor([encode(t) for t in test_texts], dtype=torch.long)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    print(f"  IMDB: {len(X_train)} train, {len(X_test)} test samples")
    return X_train, y_train, X_test, y_test, vocab_size


def load_cifar10_data(n_train=5000, n_test=1000):
    """Load CIFAR-10 from HuggingFace with proper train/test split."""
    from datasets import load_dataset
    print("  Loading CIFAR-10 dataset from HuggingFace...")
    ds = load_dataset("uoft-cs/cifar10")

    train_ds = ds["train"].shuffle(seed=SEED).select(range(n_train))
    test_ds = ds["test"].shuffle(seed=SEED).select(range(n_test))

    def process_images(split_ds):
        images, labels = [], []
        for ex in split_ds:
            img = ex["img"]
            img_array = np.array(img).astype(np.float32) / 255.0
            if img_array.ndim == 2:
                img_array = np.stack([img_array]*3, axis=-1)
            img_array = img_array.transpose(2, 0, 1)  # CHW
            images.append(img_array)
            labels.append(ex["label"])
        return torch.tensor(np.array(images), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    X_train, y_train = process_images(train_ds)
    X_test, y_test = process_images(test_ds)
    print(f"  CIFAR-10: {len(X_train)} train, {len(X_test)} test samples")
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────

class SmallTransformerClassifier(nn.Module):
    """Small transformer for sequence/token classification."""
    def __init__(self, vocab_size=5000, d_model=64, nhead=4, num_layers=2,
                 num_classes=2, max_len=128, input_type="text"):
        super().__init__()
        self.input_type = input_type
        self.d_model = d_model
        if input_type == "text":
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        else:  # vision: patch embedding
            self.patch_size = 8
            self.patch_embed = nn.Linear(3 * self.patch_size * self.patch_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.stored_attentions = []

    def forward(self, x, return_attention=False):
        if self.input_type == "text":
            x = self.embedding(x)
        else:
            B, C, H, W = x.shape
            p = self.patch_size
            x = x.unfold(2, p, p).unfold(3, p, p)
            x = x.contiguous().view(B, -1, C * p * p)
            x = self.patch_embed(x)

        B, S, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_encoding[:, :x.size(1), :]

        if return_attention:
            self.stored_attentions = []
            for layer in self.transformer.layers:
                x2, attn_weights = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
                self.stored_attentions.append(attn_weights.detach().cpu())
                x = layer.norm1(x + layer.dropout1(x2))
                x = layer.norm2(x + layer._ff_block(x))
        else:
            x = self.transformer(x)

        cls_output = x[:, 0, :]
        return self.classifier(cls_output)


class LSTMClassifier(nn.Module):
    """LSTM baseline for text."""
    def __init__(self, vocab_size=5000, embed_dim=64, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True,
                           dropout=0.1, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, return_attention=False):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.classifier(h)


class CNNClassifier(nn.Module):
    """CNN baseline for vision."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x, return_attention=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class BoWClassifier(nn.Module):
    """Bag-of-words baseline for text sentiment."""
    def __init__(self, vocab_size=5000, embed_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_attention=False):
        x = self.embedding(x)
        mask = (x.sum(-1) != 0).float().unsqueeze(-1)
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(x)


# ─────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_acc = correct / total
        val_accs.append(val_acc)
        best_val_acc = max(best_val_acc, val_acc)

    return {"best_val_acc": best_val_acc, "final_val_acc": val_acc,
            "train_losses": train_losses, "val_accs": val_accs}


def extract_attention_patterns(model, data_loader, num_samples=50):
    model.eval()
    all_attentions = []
    count = 0
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(DEVICE)
            for i in range(X_batch.size(0)):
                if count >= num_samples:
                    break
                model(X_batch[i:i+1], return_attention=True)
                attns = [a.squeeze(0).numpy() for a in model.stored_attentions]
                all_attentions.append(attns)
                count += 1
            if count >= num_samples:
                break
    return all_attentions


def compute_attention_stats(all_attentions):
    entropies = []
    max_weights = []

    for sample_attns in all_attentions:
        for layer_attn in sample_attns:
            for head in range(layer_attn.shape[0]):
                attn = layer_attn[head]
                for row in attn:
                    row = row + 1e-10
                    entropy = -np.sum(row * np.log(row))
                    entropies.append(float(entropy))
                    max_weights.append(float(np.max(row)))

    return {
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies)),
        "mean_max_weight": float(np.mean(max_weights)),
        "std_max_weight": float(np.std(max_weights)),
        "entropy_distribution": [float(e) for e in entropies[:2000]],
    }


# ─────────────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────────────

def run_experiment():
    results = {}

    # ── Load data once ──
    X_imdb_train, y_imdb_train, X_imdb_test, y_imdb_test, vocab_size = load_imdb_data()
    X_cifar_train, y_cifar_train, X_cifar_test, y_cifar_test = load_cifar10_data()

    # ── IMDB Sentiment (Text) ──
    print("\n" + "="*60)
    print("EXPERIMENT: IMDB Sentiment Classification")
    print("="*60)

    imdb_results = {"transformer": [], "lstm": [], "bow": []}
    imdb_attentions = None

    test_loader_imdb = DataLoader(TensorDataset(X_imdb_test, y_imdb_test), batch_size=64)

    for seed in range(NUM_SEEDS):
        set_seed(SEED + seed)
        train_loader = DataLoader(TensorDataset(X_imdb_train, y_imdb_train), batch_size=64, shuffle=True)

        # Transformer
        model_t = SmallTransformerClassifier(vocab_size=vocab_size, d_model=64, nhead=4,
                                             num_layers=2, num_classes=2, max_len=128, input_type="text")
        res_t = train_model(model_t, train_loader, test_loader_imdb, epochs=10, lr=5e-4)
        imdb_results["transformer"].append(res_t["best_val_acc"])
        print(f"  Seed {seed}: Transformer acc={res_t['best_val_acc']:.4f}")

        if seed == 0:
            imdb_attentions = extract_attention_patterns(model_t, test_loader_imdb)

        # LSTM
        model_l = LSTMClassifier(vocab_size=vocab_size, embed_dim=64, hidden_dim=64, num_classes=2)
        res_l = train_model(model_l, train_loader, test_loader_imdb, epochs=10, lr=5e-4)
        imdb_results["lstm"].append(res_l["best_val_acc"])
        print(f"  Seed {seed}: LSTM acc={res_l['best_val_acc']:.4f}")

        # BoW
        model_b = BoWClassifier(vocab_size=vocab_size, embed_dim=64, num_classes=2)
        res_b = train_model(model_b, train_loader, test_loader_imdb, epochs=10, lr=5e-4)
        imdb_results["bow"].append(res_b["best_val_acc"])
        print(f"  Seed {seed}: BoW acc={res_b['best_val_acc']:.4f}")

    results["imdb"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
        for k, v in imdb_results.items()
    }

    # ── CIFAR-10 (Vision) ──
    print("\n" + "="*60)
    print("EXPERIMENT: CIFAR-10 Image Classification")
    print("="*60)

    cifar_results = {"transformer": [], "cnn": []}
    cifar_attentions = None

    test_loader_cifar = DataLoader(TensorDataset(X_cifar_test, y_cifar_test), batch_size=64)

    for seed in range(NUM_SEEDS):
        set_seed(SEED + seed)
        train_loader = DataLoader(TensorDataset(X_cifar_train, y_cifar_train), batch_size=64, shuffle=True)

        # Vision Transformer
        model_vt = SmallTransformerClassifier(d_model=64, nhead=4, num_layers=2,
                                              num_classes=10, max_len=17, input_type="vision")
        res_vt = train_model(model_vt, train_loader, test_loader_cifar, epochs=30, lr=5e-4)
        cifar_results["transformer"].append(res_vt["best_val_acc"])
        print(f"  Seed {seed}: ViT acc={res_vt['best_val_acc']:.4f}")

        if seed == 0:
            cifar_attentions = extract_attention_patterns(model_vt, test_loader_cifar)

        # CNN
        model_cnn = CNNClassifier(num_classes=10)
        res_cnn = train_model(model_cnn, train_loader, test_loader_cifar, epochs=30, lr=5e-4)
        cifar_results["cnn"].append(res_cnn["best_val_acc"])
        print(f"  Seed {seed}: CNN acc={res_cnn['best_val_acc']:.4f}")

    results["cifar10"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
        for k, v in cifar_results.items()
    }

    # ── Attention Pattern Analysis ──
    print("\n" + "="*60)
    print("ANALYSIS: Cross-Domain Attention Patterns")
    print("="*60)

    attn_stats = {}
    if imdb_attentions:
        stats = compute_attention_stats(imdb_attentions)
        attn_stats["imdb"] = stats
        print(f"  IMDB attention entropy: {stats['mean_entropy']:.4f} ± {stats['std_entropy']:.4f}")
        print(f"  IMDB max attention weight: {stats['mean_max_weight']:.4f} ± {stats['std_max_weight']:.4f}")

    if cifar_attentions:
        stats = compute_attention_stats(cifar_attentions)
        attn_stats["cifar10"] = stats
        print(f"  CIFAR-10 attention entropy: {stats['mean_entropy']:.4f} ± {stats['std_entropy']:.4f}")
        print(f"  CIFAR-10 max attention weight: {stats['mean_max_weight']:.4f} ± {stats['std_max_weight']:.4f}")

    results["attention_stats"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "entropy_distribution"}
        for k, v in attn_stats.items()
    }

    # Save results
    with open(os.path.join(RESULTS_DIR, "cross_domain_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    attn_dists = {k: v.get("entropy_distribution", []) for k, v in attn_stats.items()}
    with open(os.path.join(RESULTS_DIR, "attention_distributions.json"), "w") as f:
        json.dump(attn_dists, f)

    print("\n✓ Results saved to results/cross_domain_results.json")
    return results, attn_stats


if __name__ == "__main__":
    results, attn_stats = run_experiment()
