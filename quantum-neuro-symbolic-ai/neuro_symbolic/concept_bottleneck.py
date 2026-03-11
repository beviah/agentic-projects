"""Concept Bottleneck Models Implementation.

Forces neural networks to make predictions through interpretable,
human-understandable concepts as an intermediate layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class ConceptPredictor(nn.Module):
    """Predicts concept probabilities from input features."""
    
    def __init__(self, input_dim: int, n_concepts: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.n_concepts = n_concepts
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_concepts),
            nn.Sigmoid()  # Output concept probabilities
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict concepts from input.
        
        Args:
            x: Input features of shape (batch, input_dim)
            
        Returns:
            Concept probabilities of shape (batch, n_concepts)
        """
        return self.network(x)


class LabelPredictor(nn.Module):
    """Predicts final labels from concept probabilities."""
    
    def __init__(self, n_concepts: int, n_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        
        self.network = nn.Sequential(
            nn.Linear(n_concepts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """Predict labels from concepts.
        
        Args:
            concepts: Concept probabilities of shape (batch, n_concepts)
            
        Returns:
            Class logits of shape (batch, n_classes)
        """
        return self.network(concepts)


class IndependentCBM(nn.Module):
    """Independent Concept Bottleneck Model.
    
    Each concept is predicted independently from the input.
    Architecture: Input -> Concept Predictor -> Concepts -> Label Predictor -> Output
    """
    
    def __init__(self, input_dim: int, n_concepts: int, n_classes: int,
                 concept_hidden_dim: int = 128, label_hidden_dim: int = 64,
                 concept_names: Optional[List[str]] = None):
        super().__init__()
        self.input_dim = input_dim
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.concept_names = concept_names or [f"concept_{i}" for i in range(n_concepts)]
        
        # Two-stage architecture
        self.concept_predictor = ConceptPredictor(input_dim, n_concepts, concept_hidden_dim)
        self.label_predictor = LabelPredictor(n_concepts, n_classes, label_hidden_dim)
    
    def forward(self, x: torch.Tensor, intervene_concepts: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional concept intervention.
        
        Args:
            x: Input features of shape (batch, input_dim)
            intervene_concepts: Optional concept values to force (batch, n_concepts)
                               Use -1 for concepts that should be predicted normally
            
        Returns:
            (concept_probs, class_logits) tuple
        """
        # Predict concepts
        concept_probs = self.concept_predictor(x)
        
        # Apply interventions if provided
        if intervene_concepts is not None:
            mask = (intervene_concepts >= 0).float()
            concept_probs = concept_probs * (1 - mask) + intervene_concepts * mask
        
        # Predict labels from concepts
        class_logits = self.label_predictor(concept_probs)
        
        return concept_probs, class_logits
    
    def predict_with_explanation(self, x: torch.Tensor, threshold: float = 0.5) -> Dict:
        """Make prediction with human-readable explanation.
        
        Args:
            x: Input features (single sample)
            threshold: Threshold for binary concept activation
            
        Returns:
            Dictionary with prediction and explanation
        """
        with torch.no_grad():
            concept_probs, class_logits = self.forward(x.unsqueeze(0))
            concept_probs = concept_probs.squeeze(0)
            class_logits = class_logits.squeeze(0)
            
            predicted_class = torch.argmax(class_logits).item()
            class_prob = torch.softmax(class_logits, dim=0)[predicted_class].item()
            
            # Get active concepts
            active_concepts = []
            for i, prob in enumerate(concept_probs):
                if prob > threshold:
                    active_concepts.append({
                        'name': self.concept_names[i],
                        'probability': prob.item()
                    })
            
            return {
                'predicted_class': predicted_class,
                'class_probability': class_prob,
                'active_concepts': active_concepts,
                'all_concept_probs': concept_probs.cpu().numpy()
            }


class SequentialCBM(nn.Module):
    """Sequential Concept Bottleneck Model.
    
    Concepts are predicted sequentially, allowing dependencies between concepts.
    """
    
    def __init__(self, input_dim: int, n_concepts: int, n_classes: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        
        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Sequential concept predictors (each depends on previous concepts)
        self.concept_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + i, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for i in range(n_concepts)
        ])
        
        # Label predictor
        self.label_predictor = LabelPredictor(n_concepts, n_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features of shape (batch, input_dim)
            
        Returns:
            (concept_probs, class_logits) tuple
        """
        batch_size = x.shape[0]
        
        # Encode input
        encoded = self.input_encoder(x)
        
        # Predict concepts sequentially
        concept_probs = []
        for i in range(self.n_concepts):
            # Concatenate with previously predicted concepts
            if i == 0:
                concept_input = encoded
            else:
                prev_concepts = torch.cat(concept_probs, dim=-1)
                concept_input = torch.cat([encoded, prev_concepts], dim=-1)
            
            # Predict current concept
            concept_i = self.concept_predictors[i](concept_input)
            concept_probs.append(concept_i)
        
        # Concatenate all concepts
        all_concepts = torch.cat(concept_probs, dim=-1)
        
        # Predict labels
        class_logits = self.label_predictor(all_concepts)
        
        return all_concepts, class_logits


class JointCBM(nn.Module):
    """Joint Concept Bottleneck Model.
    
    All concepts predicted jointly with shared representations.
    """
    
    def __init__(self, input_dim: int, n_concepts: int, n_classes: int,
                 shared_dim: int = 128, concept_dim: int = 64):
        super().__init__()
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU()
        )
        
        # Concept-specific heads
        self.concept_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, concept_dim),
                nn.ReLU(),
                nn.Linear(concept_dim, 1),
                nn.Sigmoid()
            )
            for _ in range(n_concepts)
        ])
        
        # Label predictor
        self.label_predictor = LabelPredictor(n_concepts, n_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features of shape (batch, input_dim)
            
        Returns:
            (concept_probs, class_logits) tuple
        """
        # Shared encoding
        shared = self.shared_encoder(x)
        
        # Predict all concepts from shared representation
        concept_probs = [head(shared) for head in self.concept_heads]
        all_concepts = torch.cat(concept_probs, dim=-1)
        
        # Predict labels
        class_logits = self.label_predictor(all_concepts)
        
        return all_concepts, class_logits


class CBMTrainer:
    """Trainer for Concept Bottleneck Models."""
    
    def __init__(self, model: nn.Module, concept_weight: float = 0.5):
        """Initialize trainer.
        
        Args:
            model: CBM model
            concept_weight: Weight for concept loss vs task loss
        """
        self.model = model
        self.concept_weight = concept_weight
    
    def compute_loss(self, concept_probs: torch.Tensor, class_logits: torch.Tensor,
                    concept_labels: torch.Tensor, class_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute joint loss.
        
        Args:
            concept_probs: Predicted concept probabilities
            class_logits: Predicted class logits
            concept_labels: True concept labels (binary)
            class_labels: True class labels
            
        Returns:
            (total_loss, loss_dict) tuple
        """
        # Concept loss (binary cross-entropy)
        concept_loss = F.binary_cross_entropy(concept_probs, concept_labels.float())
        
        # Task loss (cross-entropy)
        task_loss = F.cross_entropy(class_logits, class_labels)
        
        # Combined loss
        total_loss = self.concept_weight * concept_loss + (1 - self.concept_weight) * task_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'concept': concept_loss.item(),
            'task': task_loss.item()
        }
        
        return total_loss, loss_dict


class ConceptInterventionSimulator:
    """Simulates concept interventions for testing CBMs."""
    
    def __init__(self, model: IndependentCBM):
        self.model = model
    
    def intervene_single_concept(self, x: torch.Tensor, concept_idx: int, 
                                concept_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Intervene on a single concept.
        
        Args:
            x: Input features
            concept_idx: Index of concept to intervene on
            concept_value: Value to set (0.0 or 1.0)
            
        Returns:
            (original_prediction, intervened_prediction) tuple
        """
        with torch.no_grad():
            # Original prediction
            _, orig_logits = self.model(x)
            orig_pred = torch.argmax(orig_logits, dim=-1)
            
            # Create intervention
            batch_size = x.shape[0]
            intervention = torch.full((batch_size, self.model.n_concepts), -1.0)
            intervention[:, concept_idx] = concept_value
            
            # Intervened prediction
            _, interv_logits = self.model(x, intervene_concepts=intervention)
            interv_pred = torch.argmax(interv_logits, dim=-1)
            
            return orig_pred, interv_pred
    
    def test_concept_importance(self, x: torch.Tensor, y: torch.Tensor) -> Dict[int, float]:
        """Test importance of each concept by intervening.
        
        Args:
            x: Input features
            y: True labels
            
        Returns:
            Dictionary mapping concept_idx to importance score
        """
        importance = {}
        
        with torch.no_grad():
            # Get original predictions
            _, orig_logits = self.model(x)
            orig_correct = (torch.argmax(orig_logits, dim=-1) == y).float().mean().item()
            
            # Test each concept
            for concept_idx in range(self.model.n_concepts):
                # Try setting concept to 0
                orig_pred, pred_0 = self.intervene_single_concept(x, concept_idx, 0.0)
                acc_0 = (pred_0 == y).float().mean().item()
                
                # Try setting concept to 1
                _, pred_1 = self.intervene_single_concept(x, concept_idx, 1.0)
                acc_1 = (pred_1 == y).float().mean().item()
                
                # Importance = how much accuracy changes on average
                importance[concept_idx] = abs(acc_0 - orig_correct) + abs(acc_1 - orig_correct)
        
        return importance


if __name__ == "__main__":
    print("=" * 60)
    print("Concept Bottleneck Models Example")
    print("=" * 60)
    
    # Create synthetic dataset
    # Task: Classify animals based on concepts like "has_fur", "has_wings", "is_large"
    
    concept_names = ['has_fur', 'has_wings', 'can_fly', 'is_large', 'is_carnivore']
    class_names = ['dog', 'cat', 'bird', 'whale']
    
    # Create CBM
    cbm = IndependentCBM(
        input_dim=20,
        n_concepts=len(concept_names),
        n_classes=len(class_names),
        concept_names=concept_names
    )
    
    print(f"\nModel architecture:")
    print(f"  Input dim: 20")
    print(f"  Concepts: {len(concept_names)}")
    print(f"  Classes: {len(class_names)}")
    print(f"  Concept names: {concept_names}")
    
    # Test forward pass
    x_sample = torch.randn(3, 20)
    concepts, logits = cbm(x_sample)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {x_sample.shape}")
    print(f"  Concept predictions shape: {concepts.shape}")
    print(f"  Class logits shape: {logits.shape}")
    
    # Test prediction with explanation
    print("\n" + "=" * 60)
    print("Prediction with Explanation")
    print("=" * 60)
    
    explanation = cbm.predict_with_explanation(x_sample[0])
    
    print(f"\nPredicted class: {class_names[explanation['predicted_class']]}")
    print(f"Confidence: {explanation['class_probability']:.3f}")
    print(f"\nActive concepts:")
    for concept in explanation['active_concepts']:
        print(f"  - {concept['name']}: {concept['probability']:.3f}")
    
    # Test concept intervention
    print("\n" + "=" * 60)
    print("Concept Intervention")
    print("=" * 60)
    
    # Force "has_wings" to be true
    intervention = torch.full((1, len(concept_names)), -1.0)
    intervention[0, 1] = 1.0  # has_wings = True
    
    orig_concepts, orig_logits = cbm(x_sample[0:1])
    interv_concepts, interv_logits = cbm(x_sample[0:1], intervene_concepts=intervention)
    
    orig_class = torch.argmax(orig_logits[0]).item()
    interv_class = torch.argmax(interv_logits[0]).item()
    
    print(f"\nOriginal prediction: {class_names[orig_class]}")
    print(f"After forcing 'has_wings'=True: {class_names[interv_class]}")
    print(f"\nOriginal concepts: {orig_concepts[0].detach().numpy()}")
    print(f"Intervened concepts: {interv_concepts[0].detach().numpy()}")
    
    # Compare different CBM architectures
    print("\n" + "=" * 60)
    print("Comparing CBM Architectures")
    print("=" * 60)
    
    sequential_cbm = SequentialCBM(input_dim=20, n_concepts=5, n_classes=4)
    joint_cbm = JointCBM(input_dim=20, n_concepts=5, n_classes=4)
    
    seq_concepts, seq_logits = sequential_cbm(x_sample)
    joint_concepts, joint_logits = joint_cbm(x_sample)
    
    print(f"\nSequential CBM output shapes:")
    print(f"  Concepts: {seq_concepts.shape}")
    print(f"  Logits: {seq_logits.shape}")
    
    print(f"\nJoint CBM output shapes:")
    print(f"  Concepts: {joint_concepts.shape}")
    print(f"  Logits: {joint_logits.shape}")
    
    print("\n" + "=" * 60)
    print("All CBM implementations working correctly!")
    print("=" * 60)
