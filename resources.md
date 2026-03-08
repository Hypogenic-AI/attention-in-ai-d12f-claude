# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project: "Attention is all you need - applicable beyond transformers/LLMs to the entire generative AI and Internet field."

Resources span the computational attention mechanism (transformers, vision, audio, recommendation) and the attention economy (internet platforms, content ranking).

## Papers
Total papers downloaded: 20

| Title | Authors | Year | File | Key Domain |
|-------|---------|------|------|------------|
| Attention Is All You Need | Vaswani et al. | 2017 | papers/1706.03762_*.pdf | Foundational |
| BERT | Devlin et al. | 2019 | papers/1810.04805_*.pdf | NLP |
| General Survey on Attention | Brauwers & Frasincar | 2022 | papers/2203.14263_*.pdf | Survey (cross-domain) |
| Attention in CV Survey | Guo et al. | 2022 | papers/2111.07624_*.pdf | Computer Vision |
| Transformers Survey | - | 2023 | papers/2306.07303_*.pdf | Survey (applications) |
| ASR Deep Learning | - | 2024 | papers/2403.01255_*.pdf | Speech |
| Multimodal Alignment | - | 2024 | papers/2411.17040_*.pdf | Multimodal |
| End of Transformers? | - | 2025 | papers/2510.05364_*.pdf | Alternatives |
| Recommender Systems Review | - | 2024 | papers/2407.13699_*.pdf | Recommendation |
| Vision Transformer (ViT) | Dosovitskiy et al. | 2021 | papers/2010.11929_*.pdf | Computer Vision |
| DALL-E | Ramesh et al. | 2021 | papers/2102.12092_*.pdf | Image Generation |
| DiT | Peebles & Xie | 2023 | papers/2212.09748_*.pdf | Image Generation |
| Stable Diffusion 3 | Esser et al. | 2024 | papers/2403.03206_*.pdf | Image Generation |
| Whisper | Radford et al. | 2022 | papers/2212.04356_*.pdf | Speech |
| Music Transformer | Huang et al. | 2018 | papers/1809.04281_*.pdf | Music |
| SASRec | Kang & McAuley | 2018 | papers/1808.09781_*.pdf | Recommendation |
| Attention Calibration RecSys | - | 2023 | papers/2308.09419_*.pdf | Recommendation |
| LinkedIn Ranking | LinkedIn | 2025 | papers/2502.03417_*.pdf | Internet/Ranking |
| Graph Attention Networks | Veličković et al. | 2018 | papers/1710.10903_*.pdf | Graphs |
| AlphaFold-Multimer | DeepMind | 2021 | papers/2112.11446_*.pdf | Biology |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Domain |
|------|--------|------|------|----------|--------|
| WMT14 (EN-DE) | HuggingFace wmt14 | 500 examples | Translation | datasets/wmt14_sample/ | Text/NLP |
| CIFAR-10 | HuggingFace cifar10 | 1000 examples | Image Classification | datasets/cifar10_sample/ | Vision |
| IMDB Reviews | HuggingFace stanfordnlp/imdb | 1000 examples | Sentiment | datasets/imdb_sample/ | Text/Recommendation |

See datasets/README.md for download instructions and details.

## Code Repositories
No specific code repositories were specified in the research specification. The following libraries are recommended for experimentation:

| Library | Purpose | Install |
|---------|---------|---------|
| transformers (HuggingFace) | Pre-trained transformer models | `uv add transformers` |
| torch | PyTorch deep learning | `uv add torch` |
| datasets | Dataset loading | `uv add datasets` |
| timm | Vision transformer models | `uv add timm` |

## Resource Gathering Notes

### Search Strategy
1. Used web search across arXiv, Semantic Scholar, and academic databases
2. Searched for papers covering attention in: NLP, computer vision, speech, music, recommendation systems, graph networks, biology, and internet platforms
3. Also searched for attention economy literature connecting computational attention to human attention economics

### Selection Criteria
- Papers demonstrating attention mechanism in different domains (supporting cross-domain hypothesis)
- Foundational papers (original Transformer, ViT, BERT)
- Recent surveys providing comprehensive overviews
- Papers connecting attention to internet platform design (recommendation, ranking)
- Counterpoint papers (alternatives to attention) for balanced perspective

### Challenges Encountered
- Speech Commands dataset required legacy loading script (unavailable)
- Amazon Reviews 2023 dataset required legacy script
- Paper-finder service was unavailable; used web search as fallback

### Gaps and Workarounds
- No dedicated attention economy dataset exists; IMDB reviews serve as proxy for user engagement/recommendation domain
- Audio domain represented through papers rather than a downloadable dataset sample

## Recommendations for Experiment Design

### 1. Primary Datasets
- **WMT14** (text translation): Original transformer domain, validates baseline
- **CIFAR-10** (vision): Tests attention in vision via ViT
- **IMDB** (sentiment): Tests attention for recommendation/engagement analysis

### 2. Baseline Methods
- **LSTM/GRU**: Non-attention sequential baselines
- **CNN (ResNet)**: Non-attention vision baseline
- **Attention models**: Transformer/ViT/BERT across all domains

### 3. Evaluation Metrics
- BLEU (translation), Accuracy (classification), Attention entropy/visualization
- Cross-domain attention pattern analysis (are attention patterns similar across modalities?)

### 4. Suggested Experiments
1. **Cross-domain attention universality**: Train small transformers on text, image, and sentiment tasks; compare attention weight distributions
2. **Attention ablation**: Systematically remove/modify attention components across domains to measure impact
3. **Attention economy mapping**: Analyze how attention weights in recommendation models correlate with user engagement metrics
4. **Attention pattern visualization**: Compare attention heatmaps across modalities to identify universal patterns
