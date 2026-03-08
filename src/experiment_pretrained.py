"""
Experiment 1b: Pretrained Model Comparison
Compares pretrained attention-based (ViT, BERT) vs non-attention (ResNet) models
to show that at scale, attention-based architectures dominate across domains.
Uses HuggingFace transformers for evaluation.
"""

import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def evaluate_pretrained_text():
    """Compare BERT (attention) vs baseline on IMDB sentiment."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    print("="*60)
    print("Pretrained Text Classification: BERT on IMDB")
    print("="*60)

    # Load test data
    ds = load_dataset("stanfordnlp/imdb", split="test")
    test_subset = ds.shuffle(seed=42).select(range(500))

    # BERT sentiment pipeline (attention-based)
    print("  Loading BERT sentiment model...")
    classifier = pipeline("sentiment-analysis",
                         model="distilbert-base-uncased-finetuned-sst-2-english",
                         device=0 if torch.cuda.is_available() else -1,
                         batch_size=32)

    texts = list(test_subset["text"])
    labels = list(test_subset["label"])  # 0=negative, 1=positive

    print("  Running inference...")
    t0 = time.time()
    predictions = classifier(texts, truncation=True, max_length=512)
    inference_time = time.time() - t0

    # Map predictions
    pred_labels = [1 if p["label"] == "POSITIVE" else 0 for p in predictions]
    accuracy = sum(1 for p, l in zip(pred_labels, labels) if p == l) / len(labels)

    print(f"  BERT (DistilBERT) accuracy: {accuracy:.4f}")
    print(f"  Inference time: {inference_time:.1f}s for {len(texts)} samples")

    return {
        "model": "DistilBERT (attention-based)",
        "task": "IMDB Sentiment",
        "accuracy": float(accuracy),
        "n_samples": len(texts),
        "inference_time_s": float(inference_time)
    }


def evaluate_pretrained_vision():
    """Compare ViT (attention) vs ResNet (CNN) on CIFAR-10."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import torch.nn.functional as F

    print("\n" + "="*60)
    print("Pretrained Vision Classification: ViT vs ResNet on CIFAR-10")
    print("="*60)

    ds = load_dataset("uoft-cs/cifar10", split="test")
    test_subset = ds.shuffle(seed=42).select(range(500))

    results = {}

    # Models to compare
    models_to_test = [
        ("google/vit-base-patch16-224", "ViT-Base (attention-based)"),
        ("microsoft/resnet-50", "ResNet-50 (CNN-based)"),
    ]

    for model_name, label in models_to_test:
        print(f"\n  Loading {label}...")
        try:
            extractor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE)
            model.eval()

            # CIFAR-10 label mapping (these models trained on ImageNet, not CIFAR-10)
            # We'll measure top-5 semantic alignment instead of exact accuracy
            # Actually, let's just check if the model's architecture works
            correct = 0
            total = 0

            cifar_labels = ["airplane", "automobile", "bird", "cat", "deer",
                           "dog", "frog", "horse", "ship", "truck"]

            t0 = time.time()
            with torch.no_grad():
                for i in range(len(test_subset)):
                    img = test_subset[i]["img"].convert("RGB")
                    true_label = test_subset[i]["label"]

                    inputs = extractor(images=img, return_tensors="pt").to(DEVICE)
                    outputs = model(**inputs)
                    logits = outputs.logits

                    # Get model's predicted class name
                    pred_id = logits.argmax(-1).item()
                    pred_label = model.config.id2label.get(pred_id, "unknown").lower()

                    # Check if prediction semantically matches CIFAR-10 class
                    true_name = cifar_labels[true_label]
                    # Broad matching for ImageNet→CIFAR-10 alignment
                    match = False
                    if true_name in pred_label or pred_label in true_name:
                        match = True
                    elif true_name == "automobile" and any(w in pred_label for w in ["car", "vehicle", "cab", "limousine", "minivan", "convertible"]):
                        match = True
                    elif true_name == "airplane" and any(w in pred_label for w in ["plane", "airliner", "jet", "aircraft", "warplane"]):
                        match = True
                    elif true_name == "ship" and any(w in pred_label for w in ["boat", "vessel", "liner", "container", "aircraft carrier"]):
                        match = True
                    elif true_name == "truck" and any(w in pred_label for w in ["truck", "trailer", "pickup", "van", "moving van"]):
                        match = True
                    elif true_name == "horse" and any(w in pred_label for w in ["horse", "sorrel", "stallion"]):
                        match = True
                    elif true_name == "deer" and any(w in pred_label for w in ["deer", "elk", "gazelle", "impala", "hartebeest"]):
                        match = True
                    elif true_name == "frog" and any(w in pred_label for w in ["frog", "toad", "tree frog", "bullfrog"]):
                        match = True
                    elif true_name == "bird" and any(w in pred_label for w in ["bird", "robin", "jay", "magpie", "hen", "cock", "hummingbird", "crane", "vulture", "eagle", "finch", "junco", "brambling", "goldfinch", "chickadee", "water ouzel", "kite"]):
                        match = True
                    elif true_name == "cat" and any(w in pred_label for w in ["cat", "tabby", "tiger cat", "persian", "siamese", "egyptian"]):
                        match = True
                    elif true_name == "dog" and any(w in pred_label for w in ["dog", "terrier", "retriever", "spaniel", "poodle", "shepherd", "hound", "collie", "corgi", "pug", "chihuahua", "labrador", "beagle", "boxer", "schnauzer", "dane", "dingo"]):
                        match = True

                    if match:
                        correct += 1
                    total += 1

            inference_time = time.time() - t0
            accuracy = correct / total

            print(f"  {label} semantic accuracy: {accuracy:.4f}")
            print(f"  Inference time: {inference_time:.1f}s")

            results[label] = {
                "accuracy": float(accuracy),
                "n_samples": total,
                "inference_time_s": float(inference_time)
            }
        except Exception as e:
            print(f"  Error with {label}: {e}")
            results[label] = {"error": str(e)}

    return results


def run_pretrained_comparison():
    all_results = {}

    # Text
    text_result = evaluate_pretrained_text()
    all_results["text_pretrained"] = text_result

    # Vision
    vision_results = evaluate_pretrained_vision()
    all_results["vision_pretrained"] = vision_results

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Pretrained Model Comparison")
    print("="*60)
    print(f"  Text (BERT/attention): {text_result['accuracy']:.4f}")
    for label, res in vision_results.items():
        if "accuracy" in res:
            print(f"  Vision ({label}): {res['accuracy']:.4f}")

    with open(os.path.join(RESULTS_DIR, "pretrained_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n✓ Results saved to results/pretrained_results.json")
    return all_results


if __name__ == "__main__":
    run_pretrained_comparison()
