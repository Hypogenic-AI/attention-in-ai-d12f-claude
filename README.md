# Attention Is All You Need — Beyond Transformers

**Research investigating whether the attention mechanism is a universal computational principle across generative AI and the internet, not just a transformer-specific technique.**

## Key Findings

- **Text domain**: Transformer attention outperforms BiLSTM (+2.6 pp, p=0.046) and Bag-of-Words (+4.6 pp, p=0.004) on IMDB sentiment classification
- **Vision domain**: Pretrained ViT (attention) outperforms ResNet (CNN) by 22 percentage points on CIFAR-10 semantic accuracy
- **Mathematical universality**: Transformer attention, PageRank, and collaborative filtering are formally isomorphic — all implement `output = normalize(relevance(query, keys)) @ values`
- **Cross-domain attention patterns**: Text and vision use different attention profiles (entropy 4.6 vs 2.5 nats) but the same underlying mechanism, showing domain-adaptive information routing
- **Scale matters**: Attention models need more data than specialized architectures (ViT < CNN from scratch), but dominate at scale — explaining why attention became universal as data grew

## Project Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Pre-gathered literature review
├── resources.md                 # Resource catalog
├── src/
│   ├── experiment_cross_domain.py   # Exp 1: Train from scratch across domains
│   ├── experiment_pretrained.py     # Exp 1b: Pretrained model comparison
│   ├── experiment_math_mapping.py   # Exp 3: Mathematical isomorphism analysis
│   ├── visualization.py            # Generate all plots
│   └── statistical_analysis.py     # Statistical tests
├── results/
│   ├── cross_domain_results.json    # Raw experiment results
│   ├── pretrained_results.json      # Pretrained comparison results
│   ├── math_mapping_results.json    # Mathematical mapping results
│   ├── statistical_analysis.json    # Statistical test results
│   └── plots/                       # All visualizations
├── papers/                          # Downloaded research papers
└── datasets/                        # Pre-downloaded datasets
```

## Reproducing Results

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch transformers datasets numpy scipy matplotlib seaborn scikit-learn pillow

# Run experiments
export USER=researcher  # needed for PyTorch on some systems
python src/experiment_cross_domain.py     # ~5 min with GPU
python src/experiment_pretrained.py       # ~2 min with GPU
python src/experiment_math_mapping.py     # <1 min (CPU)
python src/visualization.py              # <1 min
python src/statistical_analysis.py       # <1 min
```

Requires: Python 3.10+, NVIDIA GPU recommended (RTX 3090 used in original experiments).

## Full Report

See [REPORT.md](REPORT.md) for the complete research report with methodology, results, analysis, and discussion.
