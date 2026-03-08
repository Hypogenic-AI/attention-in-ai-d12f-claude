# Downloaded Datasets

This directory contains datasets for cross-domain attention mechanism experiments. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: WMT14 English-German Translation (Text/NLP)

### Overview
- **Source**: HuggingFace `wmt14` (de-en config)
- **Size**: 500 test examples (sampled from 3003 test set)
- **Format**: HuggingFace Dataset
- **Task**: Machine translation (the original Transformer task)
- **License**: CC-BY-4.0

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("wmt14", "de-en", split="test[:500]")
ds.save_to_disk("datasets/wmt14_sample")
```

### Why Relevant
This is the original benchmark used in "Attention Is All You Need" (Vaswani et al., 2017). Essential for validating that attention mechanisms work in the domain they were designed for.

---

## Dataset 2: CIFAR-10 (Computer Vision)

### Overview
- **Source**: HuggingFace `cifar10`
- **Size**: 1000 test examples (sampled from 10K test set)
- **Format**: HuggingFace Dataset with PIL images
- **Task**: Image classification (10 classes)
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("cifar10", split="test[:1000]")
ds.save_to_disk("datasets/cifar10_sample")
```

### Why Relevant
Demonstrates attention mechanism applicability in computer vision (Vision Transformer). Shows the "attention is all you need" principle extends beyond NLP.

---

## Dataset 3: IMDB Reviews (Sentiment/Recommendation Domain)

### Overview
- **Source**: HuggingFace `stanfordnlp/imdb`
- **Size**: 1000 test examples (sampled from 25K test set)
- **Format**: HuggingFace Dataset
- **Task**: Binary sentiment classification
- **License**: Apache-2.0

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("stanfordnlp/imdb", split="test[:1000]")
ds.save_to_disk("datasets/imdb_sample")
```

### Why Relevant
Represents the attention economy domain - user reviews and sentiment are core to how internet platforms rank and recommend content. Attention-based models (BERT, etc.) dominate sentiment analysis.

---

## Notes
- All datasets are small samples suitable for proof-of-concept experiments
- For full-scale experiments, load complete splits from HuggingFace
- Datasets span three modalities: text translation, vision, and text sentiment - supporting the cross-domain attention hypothesis
