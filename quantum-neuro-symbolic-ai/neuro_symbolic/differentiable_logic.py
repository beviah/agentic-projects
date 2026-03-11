"""Differentiable Logic Programming Implementation.

Implements fuzzy logic operators and differentiable logic programs
that can be trained with gradient descent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Callable


class FuzzyLogicOps:
    """Differentiable fuzzy logic operators."""
    
    @staticmethod
    def t_norm_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Product t-norm for fuzzy AND."""
        return a * b
    
    @staticmethod
    def t_norm_min(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Minimum t-norm for fuzzy AND."""
        return torch.min(a, b)
    
    @staticmethod
    def t_norm_lukasiewicz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Łukasiewicz t-norm for fuzzy AND."""
        return torch.max(torch.zeros_like(a), a + b - 1)
    
    @staticmethod
    def t_conorm_probsum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Probabilistic sum t-conorm for fuzzy OR."""
        return a + b - a * b
    
    @staticmethod
    def t_conorm_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Maximum t-conorm for fuzzy OR."""
        return torch.max(a, b)
    
    @staticmethod
    def negation(a: torch.Tensor) -> torch.Tensor:
        """Fuzzy negation (NOT)."""
        return 1 - a
    
    @staticmethod
    def implication(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy implication: a -> b = NOT(a) OR b."""
        return torch.max(1 - a, b)


class DifferentiableRule(nn.Module):
    """A single differentiable logic rule.
    
    Represents: head :- body1, body2, ..., bodyN
    where the conjunction is computed using a learnable weighted t-norm.
    """
    
    def __init__(self, n_body_predicates: int, t_norm: str = 'product'):
        super().__init__()
        self.n_body = n_body_predicates
        
        # Learnable weights for body predicates
        self.weights = nn.Parameter(torch.ones(n_body_predicates))
        
        # Select t-norm
        if t_norm == 'product':
            self.t_norm = FuzzyLogicOps.t_norm_product
        elif t_norm == 'min':
            self.t_norm = FuzzyLogicOps.t_norm_min
        elif t_norm == 'lukasiewicz':
            self.t_norm = FuzzyLogicOps.t_norm_lukasiewicz
        else:
            raise ValueError(f"Unknown t-norm: {t_norm}")
    
    def forward(self, body_values: torch.Tensor) -> torch.Tensor:
        """Compute head probability from body predicate values.
        
        Args:
            body_values: Tensor of shape (batch, n_body_predicates)
            
        Returns:
            Head probability tensor of shape (batch,)
        """
        # Apply learnable weights (sigmoid for [0,1] range)
        weighted_values = body_values * torch.sigmoid(self.weights)
        
        # Compute conjunction using t-norm
        # For product t-norm, this is just the product
        if self.t_norm == FuzzyLogicOps.t_norm_product:
            result = torch.prod(weighted_values, dim=-1)
        else:
            # For other t-norms, apply pairwise
            result = weighted_values[:, 0]
            for i in range(1, self.n_body):
                result = self.t_norm(result, weighted_values[:, i])
        
        return result


class DifferentiableLogicProgram(nn.Module):
    """A collection of differentiable logic rules.
    
    Implements forward chaining: repeatedly apply rules until convergence.
    """
    
    def __init__(self, n_predicates: int, rules: List[Tuple[int, List[int]]]):
        """Initialize logic program.
        
        Args:
            n_predicates: Total number of predicates
            rules: List of (head_idx, [body_idx1, body_idx2, ...]) tuples
        """
        super().__init__()
        self.n_predicates = n_predicates
        self.rules = rules
        
        # Create differentiable rule modules
        self.rule_modules = nn.ModuleList([
            DifferentiableRule(len(body_indices))
            for _, body_indices in rules
        ])
    
    def forward(self, initial_facts: torch.Tensor, max_iterations: int = 5) -> torch.Tensor:
        """Apply forward chaining.
        
        Args:
            initial_facts: Tensor of shape (batch, n_predicates) with initial truth values
            max_iterations: Maximum forward chaining iterations
            
        Returns:
            Final predicate values after inference
        """
        batch_size = initial_facts.shape[0]
        current_facts = initial_facts.clone()
        
        for iteration in range(max_iterations):
            new_facts = current_facts.clone()
            
            # Apply each rule
            for rule_idx, (head_idx, body_indices) in enumerate(self.rules):
                # Get body predicate values
                body_values = current_facts[:, body_indices]
                
                # Apply rule to compute new head value
                new_head_value = self.rule_modules[rule_idx](body_values)
                
                # Take maximum with existing value (disjunction of rules)
                new_facts[:, head_idx] = torch.max(
                    new_facts[:, head_idx],
                    new_head_value
                )
            
            # Check convergence
            if torch.allclose(current_facts, new_facts, atol=1e-4):
                break
            
            current_facts = new_facts
        
        return current_facts


class NeuralLogicModule(nn.Module):
    """Combines neural networks with differentiable logic.
    
    Neural network predicts base predicates, then logic program derives conclusions.
    """
    
    def __init__(self, input_dim: int, n_base_predicates: int, 
                 n_derived_predicates: int, rules: List[Tuple[int, List[int]]]):
        super().__init__()
        
        self.n_base = n_base_predicates
        self.n_derived = n_derived_predicates
        self.n_total = n_base_predicates + n_derived_predicates
        
        # Neural network to predict base predicates
        self.base_predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_base_predicates),
            nn.Sigmoid()  # Base predicate probabilities
        )
        
        # Logic program for derived predicates
        self.logic_program = DifferentiableLogicProgram(self.n_total, rules)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features of shape (batch, input_dim)
            
        Returns:
            (base_predicates, all_predicates) tuple
        """
        batch_size = x.shape[0]
        
        # Predict base predicates
        base_preds = self.base_predictor(x)
        
        # Initialize all predicates (base + derived)
        all_preds = torch.zeros(batch_size, self.n_total, device=x.device)
        all_preds[:, :self.n_base] = base_preds
        
        # Apply logic program
        all_preds = self.logic_program(all_preds)
        
        return base_preds, all_preds


class ProbabilisticLogic(nn.Module):
    """Probabilistic logic using learned probability distributions."""
    
    def __init__(self, n_predicates: int, embedding_dim: int = 64):
        super().__init__()
        self.n_predicates = n_predicates
        
        # Learnable embeddings for predicates
        self.predicate_embeddings = nn.Embedding(n_predicates, embedding_dim)
        
        # Probability predictor
        self.prob_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, predicate_indices: torch.Tensor) -> torch.Tensor:
        """Predict probability for given predicates.
        
        Args:
            predicate_indices: Tensor of predicate indices
            
        Returns:
            Probability values
        """
        embeddings = self.predicate_embeddings(predicate_indices)
        probs = self.prob_predictor(embeddings).squeeze(-1)
        return probs
    
    def probabilistic_and(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Probabilistic AND (assuming independence)."""
        return p1 * p2
    
    def probabilistic_or(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Probabilistic OR (assuming independence)."""
        return p1 + p2 - p1 * p2


if __name__ == "__main__":
    # Example: Family relationships
    print("=" * 60)
    print("Differentiable Logic Programming Example")
    print("=" * 60)
    
    # Define predicates:
    # 0: mother(X,Y), 1: father(X,Y), 2: parent(X,Y), 3: ancestor(X,Y)
    
    # Define rules:
    # parent(X,Y) :- mother(X,Y)  (rule 0: head=2, body=[0])
    # parent(X,Y) :- father(X,Y)  (rule 1: head=2, body=[1])
    # ancestor(X,Y) :- parent(X,Y) (rule 2: head=3, body=[2])
    
    rules = [
        (2, [0]),  # parent :- mother
        (2, [1]),  # parent :- father  
        (3, [2]),  # ancestor :- parent
    ]
    
    # Create logic program
    logic_program = DifferentiableLogicProgram(n_predicates=4, rules=rules)
    
    # Initial facts: mother=0.9, father=0.0 for person pair
    initial_facts = torch.tensor([[0.9, 0.0, 0.0, 0.0]])  # batch_size=1
    
    print("\nInitial facts:")
    print(f"  mother: {initial_facts[0, 0]:.3f}")
    print(f"  father: {initial_facts[0, 1]:.3f}")
    print(f"  parent: {initial_facts[0, 2]:.3f}")
    print(f"  ancestor: {initial_facts[0, 3]:.3f}")
    
    # Apply forward chaining
    derived_facts = logic_program(initial_facts)
    
    print("\nDerived facts after forward chaining:")
    print(f"  mother: {derived_facts[0, 0]:.3f}")
    print(f"  father: {derived_facts[0, 1]:.3f}")
    print(f"  parent: {derived_facts[0, 2]:.3f}")
    print(f"  ancestor: {derived_facts[0, 3]:.3f}")
    
    print("\n" + "=" * 60)
    print("Neural-Logic Integration Example")
    print("=" * 60)
    
    # Neural-logic module
    neural_logic = NeuralLogicModule(
        input_dim=10,
        n_base_predicates=2,  # mother, father (predicted from features)
        n_derived_predicates=2,  # parent, ancestor (derived by logic)
        rules=rules
    )
    
    # Random input
    x = torch.randn(4, 10)
    
    base_preds, all_preds = neural_logic(x)
    
    print("\nSample predictions for 4 person pairs:")
    for i in range(4):
        print(f"\nPair {i+1}:")
        print(f"  mother (base): {all_preds[i, 0]:.3f}")
        print(f"  father (base): {all_preds[i, 1]:.3f}")
        print(f"  parent (derived): {all_preds[i, 2]:.3f}")
        print(f"  ancestor (derived): {all_preds[i, 3]:.3f}")
