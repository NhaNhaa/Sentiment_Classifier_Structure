"""Pytest tests for data_loader module."""

import pytest
from data_loader import (
    load_imdb_dataset,
    get_class_distribution,
    print_dataset_summary
)


def test_load_imdb_dataset_small():
    """Test loading with small sample limit."""
    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset(max_samples=10)

    # Check we got the right number of samples
    assert len(train_texts) == 10
    assert len(train_labels) == 10
    assert len(test_texts) == 10
    assert len(test_labels) == 10

    # Check labels are 0 or 1
    for label in train_labels:
        assert label in [0, 1]

    # Check texts are strings and not empty
    for text in train_texts:
        assert isinstance(text, str)
        assert len(text) > 0


def test_load_imdb_dataset_no_limit():
    """Test loading full dataset (may take a moment)."""
    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset(max_samples=None)

    # Full IMDB has 25K train, 25K test
    assert len(train_texts) == 25000
    assert len(test_texts) == 25000


def test_get_class_distribution_normal():
    """Test class distribution counts."""
    labels = [1, 0, 1, 1, 0]
    pos, neg, total = get_class_distribution(labels)

    assert pos == 3
    assert neg == 2
    assert total == 5


def test_get_class_distribution_empty():
    """Test empty label list returns zeros."""
    pos, neg, total = get_class_distribution([])
    assert pos == 0
    assert neg == 0
    assert total == 0


def test_print_dataset_summary_runs(capsys):
    """Test that print_dataset_summary executes without error (capture output)."""
    train_texts = ["good movie", "bad movie"]
    train_labels = [1, 0]
    test_texts = ["great", "terrible"]
    test_labels = [1, 0]

    print_dataset_summary(train_texts, train_labels, test_texts, test_labels)

    captured = capsys.readouterr()
    assert "Dataset Summary" in captured.out
    assert "POSITIVE" in captured.out or "NEGATIVE" in captured.out