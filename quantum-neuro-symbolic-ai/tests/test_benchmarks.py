"""Tests for benchmark suite."""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.simple_benchmark import (
    SyntheticRelationalDataset,
    SimpleFFN,
    ConceptBaseline,
    train_model,
    evaluate_model
)


def test_dataset_generation():
    """Test dataset generates correct shapes and labels."""
    dataset = SyntheticRelationalDataset(n_samples=100, n_features=10, seed=42)
    
    assert dataset.X.shape == (100, 10), f"Expected shape (100, 10), got {dataset.X.shape}"
    assert dataset.y.shape == (100,), f"Expected shape (100,), got {dataset.y.shape}"
    assert dataset.y.dtype == torch.long, f"Expected dtype torch.long, got {dataset.y.dtype}"
    
    # Check label range
    unique_labels = torch.unique(dataset.y)
    assert len(unique_labels) <= 3, f"Expected at most 3 classes, got {len(unique_labels)}"
    assert all(label in [0, 1, 2] for label in unique_labels), "Labels must be in {0, 1, 2}"
    
    print("✓ Dataset generation test passed")


def test_dataset_label_logic():
    """Test dataset labels follow logical rules."""
    dataset = SyntheticRelationalDataset(n_samples=100, n_features=10, seed=42)
    
    # Manually check a few samples
    for i in range(10):
        x = dataset.X[i]
        y = dataset.y[i]
        
        if x[0] > 0.5 and x[1] > 0.5:
            assert y == 0, f"Sample {i}: Expected class 0, got {y}"
        elif x[2] > 0.5 and x[3] > 0.5:
            assert y == 1, f"Sample {i}: Expected class 1, got {y}"
        else:
            assert y == 2, f"Sample {i}: Expected class 2, got {y}"
    
    print("✓ Dataset label logic test passed")


def test_dataset_splits():
    """Test dataset splitting produces correct sizes."""
    dataset = SyntheticRelationalDataset(n_samples=100, n_features=10, seed=42)
    X_train, y_train, X_test, y_test = dataset.get_splits(train_ratio=0.8)
    
    assert X_train.shape[0] == 80, f"Expected 80 train samples, got {X_train.shape[0]}"
    assert X_test.shape[0] == 20, f"Expected 20 test samples, got {X_test.shape[0]}"
    assert X_train.shape[1] == 10, f"Expected 10 features, got {X_train.shape[1]}"
    assert y_train.shape[0] == 80, f"Expected 80 train labels, got {y_train.shape[0]}"
    assert y_test.shape[0] == 20, f"Expected 20 test labels, got {y_test.shape[0]}"
    
    print("✓ Dataset splits test passed")


def test_simple_ffn_forward():
    """Test SimpleFFN forward pass produces correct output shape."""
    model = SimpleFFN(input_dim=10, n_classes=3)
    x = torch.randn(5, 10)
    
    output = model(x)
    
    assert output.shape == (5, 3), f"Expected shape (5, 3), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    print("✓ SimpleFFN forward test passed")


def test_concept_baseline_forward():
    """Test ConceptBaseline forward pass produces correct output shapes."""
    model = ConceptBaseline(input_dim=10, n_concepts=4, n_classes=3)
    x = torch.randn(5, 10)
    
    logits, concepts = model(x)
    
    assert logits.shape == (5, 3), f"Expected logits shape (5, 3), got {logits.shape}"
    assert concepts.shape == (5, 4), f"Expected concepts shape (5, 4), got {concepts.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN"
    assert not torch.isnan(concepts).any(), "Concepts contain NaN"
    
    # Concepts should be in [0, 1] due to sigmoid
    assert (concepts >= 0).all() and (concepts <= 1).all(), "Concepts must be in [0, 1]"
    
    print("✓ ConceptBaseline forward test passed")


def test_training_reduces_loss():
    """Test that training actually reduces loss."""
    torch.manual_seed(42)
    dataset = SyntheticRelationalDataset(n_samples=100, n_features=10, seed=42)
    X_train, y_train, _, _ = dataset.get_splits(train_ratio=0.8)
    
    model = SimpleFFN(input_dim=10, n_classes=3)
    history = train_model(model, X_train, y_train, epochs=20, lr=0.01, verbose=False)
    
    # Check loss decreases
    initial_loss = history['loss'][0]
    final_loss = history['loss'][-1]
    
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    assert final_loss < 1.5, f"Final loss too high: {final_loss:.4f}"
    
    # Check accuracy increases
    initial_acc = history['accuracy'][0]
    final_acc = history['accuracy'][-1]
    
    assert final_acc > initial_acc or final_acc > 0.5, "Accuracy should improve or exceed random"
    
    print("✓ Training reduces loss test passed")


def test_evaluation_produces_valid_metrics():
    """Test evaluation produces valid accuracy metrics."""
    torch.manual_seed(42)
    dataset = SyntheticRelationalDataset(n_samples=100, n_features=10, seed=42)
    X_train, y_train, X_test, y_test = dataset.get_splits(train_ratio=0.8)
    
    model = SimpleFFN(input_dim=10, n_classes=3)
    train_model(model, X_train, y_train, epochs=10, verbose=False)
    
    results = evaluate_model(model, X_test, y_test)
    
    # Check accuracy is valid
    assert 'accuracy' in results, "Results must contain 'accuracy'"
    assert 0.0 <= results['accuracy'] <= 1.0, f"Accuracy must be in [0, 1], got {results['accuracy']}"
    
    # Check per-class accuracy
    assert 'per_class_accuracy' in results, "Results must contain 'per_class_accuracy'"
    for class_id, acc in results['per_class_accuracy'].items():
        assert 0.0 <= acc <= 1.0, f"Class {class_id} accuracy must be in [0, 1], got {acc}"
    
    # Check predictions
    assert 'predictions' in results, "Results must contain 'predictions'"
    assert len(results['predictions']) == len(y_test), "Predictions length mismatch"
    
    print("✓ Evaluation metrics test passed")


def test_concept_baseline_learns():
    """Test ConceptBaseline can learn the task."""
    torch.manual_seed(42)
    dataset = SyntheticRelationalDataset(n_samples=200, n_features=10, seed=42)
    X_train, y_train, X_test, y_test = dataset.get_splits(train_ratio=0.8)
    
    model = ConceptBaseline(input_dim=10, n_concepts=4, n_classes=3)
    history = train_model(model, X_train, y_train, epochs=30, lr=0.01, verbose=False)
    results = evaluate_model(model, X_test, y_test)
    
    # Should achieve reasonable accuracy
    assert results['accuracy'] > 0.4, f"ConceptBaseline accuracy too low: {results['accuracy']:.4f}"
    
    # Loss should decrease significantly
    assert history['loss'][-1] < history['loss'][0] * 0.7, "Loss reduction insufficient"
    
    print("✓ ConceptBaseline learning test passed")


if __name__ == "__main__":
    print("Running benchmark tests...\n")
    
    test_dataset_generation()
    test_dataset_label_logic()
    test_dataset_splits()
    test_simple_ffn_forward()
    test_concept_baseline_forward()
    test_training_reduces_loss()
    test_evaluation_produces_valid_metrics()
    test_concept_baseline_learns()
    
    print("\n" + "="*60)
    print("All benchmark tests passed!")
    print("="*60)
