"""Load and preprocess the IMDB dataset for sentiment classification."""

import numpy as np
from datasets import load_dataset
from config import TEST_SIZE, RANDOM_STATE, MAX_SAMPLES


def load_imdb_dataset(max_samples=MAX_SAMPLES):
    """Load IMDB dataset from Hugging Face and return train/test splits.

    Args:
        max_samples: Maximum number of samples to use per split.
                    Use None to load all 25K per split.

    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels)
        Each texts is list of strings, labels list of ints (0=neg, 1=pos).
    """
    print("Loading IMDB dataset from Hugging Face...")

    try:
        dataset = load_dataset("imdb", trust_remote_code=False)
    except Exception as error:
        print(f"ERROR: Failed to load IMDB dataset: {error}")
        return [], [], [], []

    # Extract texts and labels from train split
    train_data = dataset["train"]
    test_data = dataset["test"]

    train_texts = train_data["text"]
    train_labels = train_data["label"]
    test_texts = test_data["text"]
    test_labels = test_data["label"]

    # Apply sample limit if specified
    if max_samples is not None and max_samples > 0:
        print(f"Limiting each split to {max_samples} samples")
        train_texts = train_texts[:max_samples]
        train_labels = train_labels[:max_samples]
        test_texts = test_texts[:max_samples]
        test_labels = test_labels[:max_samples]

    print(f"Train size: {len(train_texts)}")
    print(f"Test size: {len(test_texts)}")

    return train_texts, train_labels, test_texts, test_labels


def get_class_distribution(labels):
    """Count number of positive (1) and negative (0) samples.

    Args:
        labels: List or array of integer labels.

    Returns:
        Tuple of (positive_count, negative_count, total_count)
    """
    if not labels:
        print("WARNING: Empty label list")
        return 0, 0, 0

    labels_array = np.array(labels)
    pos_count = int(np.sum(labels_array == 1))
    neg_count = int(np.sum(labels_array == 0))
    total = len(labels)

    return pos_count, neg_count, total


def print_dataset_summary(train_texts, train_labels, test_texts, test_labels):
    """Print a readable summary of dataset statistics.

    Args:
        train_texts: List of training texts.
        train_labels: List of training labels.
        test_texts: List of test texts.
        test_labels: List of test labels.
    """
    print("\n=== Dataset Summary ===")

    pos_train, neg_train, total_train = get_class_distribution(train_labels)
    pos_test, neg_test, total_test = get_class_distribution(test_labels)

    print(f"Training: {total_train} reviews")
    print(f"  Positive: {pos_train} ({pos_train/total_train*100:.1f}%)")
    print(f"  Negative: {neg_train} ({neg_train/total_train*100:.1f}%)")

    print(f"Test: {total_test} reviews")
    print(f"  Positive: {pos_test} ({pos_test/total_test*100:.1f}%)")
    print(f"  Negative: {neg_test} ({neg_test/total_test*100:.1f}%)")

    # Show first example text snippet
    if train_texts and len(train_texts) > 0:
        sample_text = train_texts[0]
        sample_label = "POSITIVE" if train_labels[0] == 1 else "NEGATIVE"
        preview = sample_text[:200] + "..." if len(sample_text) > 200 else sample_text
        print(f"\nFirst training example:")
        print(f"  Label: {sample_label}")
        print(f"  Preview: {preview}")


if __name__ == "__main__":
    print("Testing data loader...")

    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset(max_samples=100)

    if train_texts:
        print_dataset_summary(train_texts, train_labels, test_texts, test_labels)
    else:
        print("Failed to load dataset.")