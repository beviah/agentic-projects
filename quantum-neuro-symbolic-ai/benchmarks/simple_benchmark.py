"""Simple benchmark for quantum neuro-symbolic models.

Provides:
- Synthetic relational reasoning task
- Classical baseline models
- Training and evaluation pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict
import time


class SyntheticRelationalDataset:
    """Synthetic dataset for relational reasoning.
    
    Task: Learn logical rule (f0 > 0.5 AND f1 > 0.5) -> class 0
                             (f2 > 0.5 AND f3 > 0.5) -> class 1
                             else -> class 2
    """
    
    def __init__(self, n_samples: int = 1000, n_features: int = 10, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.n_samples = n_samples
        self.n_features = n_features
        
        # Generate features
        self.X = torch.randn(n_samples, n_features)
        
        # Generate labels based on logical rules
        self.y = self._generate_labels(self.X)
    
    def _generate_labels(self, X: torch.Tensor) -> torch.Tensor:
        """Generate labels based on logical rules."""
        labels = torch.zeros(X.shape[0], dtype=torch.long)
        
        for i in range(X.shape[0]):
            if X[i, 0] > 0.5 and X[i, 1] > 0.5:
                labels[i] = 0
            elif X[i, 2] > 0.5 and X[i, 3] > 0.5:
                labels[i] = 1
            else:
                labels[i] = 2
        
        return labels
    
    def get_splits(self, train_ratio: float = 0.8) -> Tuple:
        """Split into train/test sets."""
        n_train = int(self.n_samples * train_ratio)
        
        indices = torch.randperm(self.n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        return (
            self.X[train_idx], self.y[train_idx],
            self.X[test_idx], self.y[test_idx]
        )


class SimpleFFN(nn.Module):
    """Simple 3-layer feedforward baseline."""
    
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ConceptBaseline(nn.Module):
    """Classical CBM baseline."""
    
    def __init__(self, input_dim: int, n_concepts: int, n_classes: int):
        super().__init__()
        
        # Concept predictor
        self.concept_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_concepts),
            nn.Sigmoid()
        )
        
        # Class predictor from concepts
        self.class_predictor = nn.Sequential(
            nn.Linear(n_concepts, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        concepts = self.concept_predictor(x)
        logits = self.class_predictor(concepts)
        return logits, concepts


def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                epochs: int = 50, lr: float = 0.001, verbose: bool = True) -> Dict:
    """Train a model and return metrics."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, ConceptBaseline):
            logits, _ = model(X_train)
        else:
            logits = model(X_train)
        
        loss = criterion(logits, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_train).float().mean().item()
        
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={loss.item():.4f}, Acc={accuracy:.4f}")
    
    train_time = time.time() - start_time
    history['train_time'] = train_time
    
    return history


def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict:
    """Evaluate model on test set."""
    model.eval()
    
    with torch.no_grad():
        if isinstance(model, ConceptBaseline):
            logits, concepts = model(X_test)
        else:
            logits = model(X_test)
            concepts = None
        
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y_test).float().mean().item()
        
        # Per-class accuracy
        per_class_acc = {}
        for c in range(3):
            mask = y_test == c
            if mask.sum() > 0:
                class_acc = (preds[mask] == y_test[mask]).float().mean().item()
                per_class_acc[c] = class_acc
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'predictions': preds.numpy(),
        'concepts': concepts.numpy() if concepts is not None else None
    }


def run_benchmark(verbose: bool = True) -> Dict:
    """Run full benchmark suite."""
    if verbose:
        print("="*60)
        print("Simple Relational Reasoning Benchmark")
        print("="*60)
    
    # Generate dataset
    if verbose:
        print("\nGenerating synthetic dataset...")
    dataset = SyntheticRelationalDataset(n_samples=1000, n_features=10)
    X_train, y_train, X_test, y_test = dataset.get_splits(train_ratio=0.8)
    
    if verbose:
        print(f"  Train samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(torch.unique(y_train))}")
    
    results = {}
    
    # Baseline 1: Simple FFN
    if verbose:
        print("\n" + "="*60)
        print("Model 1: Simple Feedforward Network")
        print("="*60)
    
    ffn = SimpleFFN(input_dim=10, n_classes=3)
    ffn_history = train_model(ffn, X_train, y_train, epochs=50, verbose=verbose)
    ffn_results = evaluate_model(ffn, X_test, y_test)
    
    results['simple_ffn'] = {
        'train_history': ffn_history,
        'test_results': ffn_results
    }
    
    if verbose:
        print(f"\nTest Accuracy: {ffn_results['accuracy']:.4f}")
        print(f"Training Time: {ffn_history['train_time']:.2f}s")
    
    # Baseline 2: Concept Bottleneck
    if verbose:
        print("\n" + "="*60)
        print("Model 2: Classical Concept Bottleneck")
        print("="*60)
    
    cbm = ConceptBaseline(input_dim=10, n_concepts=4, n_classes=3)
    cbm_history = train_model(cbm, X_train, y_train, epochs=50, verbose=verbose)
    cbm_results = evaluate_model(cbm, X_test, y_test)
    
    results['concept_baseline'] = {
        'train_history': cbm_history,
        'test_results': cbm_results
    }
    
    if verbose:
        print(f"\nTest Accuracy: {cbm_results['accuracy']:.4f}")
        print(f"Training Time: {cbm_history['train_time']:.2f}s")
    
    # Summary
    if verbose:
        print("\n" + "="*60)
        print("Benchmark Summary")
        print("="*60)
        print(f"\nSimple FFN:")
        print(f"  Test Accuracy: {ffn_results['accuracy']:.4f}")
        print(f"  Train Time: {ffn_history['train_time']:.2f}s")
        print(f"\nConcept Baseline:")
        print(f"  Test Accuracy: {cbm_results['accuracy']:.4f}")
        print(f"  Train Time: {cbm_history['train_time']:.2f}s")
        print("\n" + "="*60)
        print("Benchmark establishes minimum bar: 70-90% accuracy")
        print("Quantum models must match or exceed this performance")
        print("="*60)
    
    return results


if __name__ == "__main__":
    results = run_benchmark(verbose=True)
