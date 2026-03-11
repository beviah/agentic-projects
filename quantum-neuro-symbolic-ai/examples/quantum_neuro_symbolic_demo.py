"""Quantum Neuro-Symbolic AI Demo.

Complete example demonstrating quantum circuits implementing differentiable
logic on knowledge graph structures.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn

try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Some features will be limited.")

from neuro_symbolic.differentiable_logic import DifferentiableLogicProgram, FuzzyLogicOps
from neuro_symbolic.knowledge_guided_nn import KnowledgeGraph, KGGuidedGNN
from neuro_symbolic.concept_bottleneck import IndependentCBM

if QISKIT_AVAILABLE:
    from quantum_ml.hybrid_quantum_classical import HybridQuantumClassicalModel
    from quantum_ml.quantum_kernels import QuantumKernel, QuantumSVM
    from quantum_neuro_symbolic.quantum_logic_circuits import QuantumLogicProgram
    from quantum_neuro_symbolic.quantum_kg_embedding import QuantumKGEmbedding


class QuantumNeuroSymbolicSystem:
    """Integrated quantum neuro-symbolic AI system.
    
    Combines:
    1. Knowledge graph structure
    2. Differentiable logic reasoning
    3. Quantum computing for logic operations
    4. Concept bottleneck for interpretability
    """
    
    def __init__(self, n_entities: int, n_relations: int, n_concepts: int,
                 n_qubits: int = 4, use_quantum: bool = True):
        """Initialize the system.
        
        Args:
            n_entities: Number of entities in knowledge graph
            n_relations: Number of relation types
            n_concepts: Number of interpretable concepts
            n_qubits: Qubits for quantum components
            use_quantum: Whether to use quantum components
        """
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_concepts = n_concepts
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum and QISKIT_AVAILABLE
        
        # 1. Knowledge Graph
        self.kg = KnowledgeGraph(n_entities, n_relations)
        
        # 2. Classical differentiable logic
        self.classical_logic = None
        
        # 3. Quantum components (if available)
        if self.use_quantum:
            self.quantum_kg = QuantumKGEmbedding(n_entities, n_relations, n_qubits)
            self.quantum_logic = None
        
        # 4. Concept bottleneck
        self.cbm = IndependentCBM(
            input_dim=n_qubits,
            n_concepts=n_concepts,
            n_classes=n_entities,
            concept_names=[f"concept_{i}" for i in range(n_concepts)]
        )
    
    def add_knowledge(self, head: int, relation: int, tail: int):
        """Add a triple to the knowledge graph."""
        self.kg.add_triple(head, relation, tail)
    
    def setup_logic_rules(self, rules: list):
        """Setup logic rules for reasoning.
        
        Args:
            rules: List of (head_idx, body_indices) tuples
        """
        # Classical logic
        n_predicates = self.n_entities + self.n_concepts
        self.classical_logic = DifferentiableLogicProgram(n_predicates, rules)
        
        # Quantum logic (if available)
        if self.use_quantum:
            quantum_rules = [(h, b, 'implication') for h, b in rules]
            self.quantum_logic = QuantumLogicProgram(n_predicates, quantum_rules)
    
    def reason_classical(self, initial_facts: np.ndarray) -> np.ndarray:
        """Perform classical logic reasoning.
        
        Args:
            initial_facts: Initial predicate values
            
        Returns:
            Derived facts after reasoning
        """
        if self.classical_logic is None:
            raise ValueError("Logic rules not setup. Call setup_logic_rules() first.")
        
        facts_tensor = torch.tensor([initial_facts], dtype=torch.float32)
        derived = self.classical_logic(facts_tensor)
        
        return derived[0].detach().numpy()
    
    def reason_quantum(self, initial_facts: np.ndarray) -> np.ndarray:
        """Perform quantum logic reasoning.
        
        Args:
            initial_facts: Initial predicate values
            
        Returns:
            Derived facts after quantum reasoning
        """
        if not self.use_quantum:
            raise ValueError("Quantum components not available")
        
        if self.quantum_logic is None:
            raise ValueError("Quantum logic rules not setup")
        
        # Random parameters for demo
        params = np.random.randn(self.quantum_logic.n_params) * 0.1
        
        derived = self.quantum_logic.forward(initial_facts, params)
        
        return derived
    
    def predict_with_explanation(self, entity_features: np.ndarray) -> dict:
        """Make prediction with interpretable explanation.
        
        Args:
            entity_features: Input features
            
        Returns:
            Dictionary with prediction and explanation
        """
        features_tensor = torch.tensor(entity_features, dtype=torch.float32)
        return self.cbm.predict_with_explanation(features_tensor)
    
    def kg_reasoning(self, head: int, relation: int) -> list:
        """Perform KG reasoning to predict tail entities.
        
        Args:
            head: Head entity
            relation: Relation type
            
        Returns:
            List of (entity, score) predictions
        """
        if self.use_quantum:
            return self.quantum_kg.predict_tail(head, relation, top_k=5)
        else:
            # Classical fallback
            neighbors = self.kg.get_neighbors(head, relation)
            return [(n, 1.0) for n in neighbors]


def demo_neuro_symbolic():
    """Demonstrate classical neuro-symbolic AI."""
    print("\n" + "=" * 70)
    print("NEURO-SYMBOLIC AI DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. Differentiable Logic Programming")
    print("-" * 70)
    
    # Define logic rules for family relationships
    # Predicates: 0=mother, 1=father, 2=parent, 3=ancestor
    rules = [
        (2, [0]),  # parent :- mother
        (2, [1]),  # parent :- father
        (3, [2]),  # ancestor :- parent
    ]
    
    logic = DifferentiableLogicProgram(n_predicates=4, rules=rules)
    
    # Initial facts
    initial = torch.tensor([[0.9, 0.1, 0.0, 0.0]], dtype=torch.float32)
    
    print(f"Initial facts: mother=0.9, father=0.1, parent=0.0, ancestor=0.0")
    
    # Forward reasoning
    derived = logic(initial)
    
    print(f"Derived facts: mother={derived[0,0]:.2f}, father={derived[0,1]:.2f}, " +
          f"parent={derived[0,2]:.2f}, ancestor={derived[0,3]:.2f}")
    
    print("\n2. Knowledge-Guided GNN")
    print("-" * 70)
    
    # Create knowledge graph
    kg = KnowledgeGraph(n_entities=6, n_relations=2)
    kg.add_triple(0, 0, 3)  # Dog is-a Mammal
    kg.add_triple(1, 0, 3)  # Cat is-a Mammal
    kg.add_triple(3, 0, 2)  # Mammal is-a Animal
    
    print(f"Knowledge graph: {len(kg.triples)} triples")
    
    # Create GNN
    gnn = KGGuidedGNN(n_entities=6, n_relations=2, embedding_dim=16, hidden_dim=32)
    
    # Forward pass
    embeddings = gnn(kg)
    
    print(f"Generated entity embeddings: shape={embeddings.shape}")
    print(f"  Dog embedding norm: {torch.norm(embeddings[0]):.3f}")
    print(f"  Cat embedding norm: {torch.norm(embeddings[1]):.3f}")
    
    print("\n3. Concept Bottleneck Model")
    print("-" * 70)
    
    concept_names = ['has_fur', 'has_wings', 'can_fly', 'is_large']
    cbm = IndependentCBM(
        input_dim=10,
        n_concepts=4,
        n_classes=3,
        concept_names=concept_names
    )
    
    # Sample input
    x = torch.randn(1, 10)
    
    explanation = cbm.predict_with_explanation(x[0])
    
    print(f"Predicted class: {explanation['predicted_class']}")
    print(f"Confidence: {explanation['class_probability']:.3f}")
    print(f"Active concepts: {len(explanation['active_concepts'])}")
    for concept in explanation['active_concepts']:
        print(f"  - {concept['name']}: {concept['probability']:.3f}")


def demo_quantum_ml():
    """Demonstrate quantum machine learning."""
    if not QISKIT_AVAILABLE:
        print("\nQuantum ML demo skipped (Qiskit not available)")
        return
    
    print("\n" + "=" * 70)
    print("QUANTUM MACHINE LEARNING DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. Hybrid Quantum-Classical Model")
    print("-" * 70)
    
    model = HybridQuantumClassicalModel(n_qubits=4, n_layers=2, n_classes=3)
    
    print(f"Model: {model.n_qubits} qubits, {model.n_layers} layers")
    print(f"Quantum parameters: {len(model.params)}")
    
    x_test = np.random.randn(4)
    logits = model.forward(x_test)
    
    print(f"Input shape: {x_test.shape}")
    print(f"Output logits: {logits}")
    print(f"Predicted class: {np.argmax(logits)}")
    
    print("\n2. Quantum Kernel SVM")
    print("-" * 70)
    
    qkernel = QuantumKernel(feature_map='zz', n_qubits=4, depth=1)
    
    # Training data
    X_train = np.array([
        [0.1, 0.1, 0.0, 0.0],
        [0.2, 0.2, 0.1, 0.1],
        [0.8, 0.8, 0.9, 0.9],
        [0.9, 0.9, 1.0, 1.0]
    ])
    y_train = np.array([-1, -1, 1, 1])
    
    print(f"Training QSVM on {len(X_train)} samples...")
    
    qsvm = QuantumSVM(qkernel, C=1.0)
    qsvm.fit(X_train, y_train)
    
    train_acc = qsvm.score(X_train, y_train)
    print(f"Training accuracy: {train_acc:.2%}")


def demo_quantum_neuro_symbolic():
    """Demonstrate integrated quantum neuro-symbolic AI."""
    if not QISKIT_AVAILABLE:
        print("\nQuantum neuro-symbolic demo skipped (Qiskit not available)")
        return
    
    print("\n" + "=" * 70)
    print("QUANTUM NEURO-SYMBOLIC AI DEMONSTRATION")
    print("=" * 70)
    
    print("\nIntegrating: Quantum Computing + Logic + Knowledge Graphs")
    
    print("\n1. Quantum Knowledge Graph Embedding")
    print("-" * 70)
    
    qkg = QuantumKGEmbedding(n_entities=6, n_relations=3, n_qubits=4)
    
    print(f"Quantum KG: {qkg.n_entities} entities, {qkg.n_relations} relations")
    print(f"Each entity encoded in {qkg.n_qubits} qubits")
    
    # Score triples
    triples = [(0, 0, 1), (0, 0, 5), (2, 1, 3)]
    
    print("\nTriple scores:")
    for h, r, t in triples:
        score = qkg.compute_triple_score(h, r, t)
        print(f"  ({h}, {r}, {t}): {score:.4f}")
    
    # Predict tail entities
    predictions = qkg.predict_tail(head=0, relation=0, top_k=3)
    
    print("\nTail predictions for (0, 0, ?):")
    for rank, (entity, score) in enumerate(predictions, 1):
        print(f"  {rank}. Entity {entity}: {score:.4f}")
    
    print("\n2. Quantum Logic Reasoning")
    print("-" * 70)
    
    # Define quantum logic rules
    rules = [
        (2, [0], 'implication'),  # parent :- mother
        (2, [1], 'implication'),  # parent :- father
    ]
    
    qlogic = QuantumLogicProgram(n_predicates=3, rules=rules)
    
    print(f"Quantum logic program: {len(rules)} rules")
    print(f"Circuit uses {qlogic.n_qubits} qubits")
    
    # Execute reasoning
    initial_state = np.array([0.9, 0.1, 0.0])
    params = np.array([np.pi/3, np.pi/3])
    
    print(f"\nInitial: mother=0.9, father=0.1, parent=0.0")
    
    final_state = qlogic.forward(initial_state, params)
    
    print(f"After quantum reasoning: mother={final_state[0]:.2f}, " +
          f"father={final_state[1]:.2f}, parent={final_state[2]:.2f}")
    
    print("\n3. Full Quantum Neuro-Symbolic System")
    print("-" * 70)
    
    system = QuantumNeuroSymbolicSystem(
        n_entities=5,
        n_relations=2,
        n_concepts=3,
        n_qubits=4,
        use_quantum=True
    )
    
    # Add knowledge
    system.add_knowledge(0, 0, 1)
    system.add_knowledge(1, 0, 2)
    system.add_knowledge(2, 1, 3)
    
    print(f"System initialized with {len(system.kg.triples)} KG triples")
    
    # Setup logic
    logic_rules = [(3, [0]), (4, [1])]
    system.setup_logic_rules(logic_rules)
    
    print(f"Logic rules configured: {len(logic_rules)} rules")
    
    # Classical reasoning
    initial_facts = np.array([0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    classical_result = system.reason_classical(initial_facts)
    
    print(f"\nClassical reasoning result: {classical_result[:5]}")
    
    # Quantum reasoning
    quantum_result = system.reason_quantum(initial_facts)
    
    print(f"Quantum reasoning result: {quantum_result[:5]}")
    
    # KG reasoning
    kg_predictions = system.kg_reasoning(head=0, relation=0)
    
    print(f"\nKG reasoning predictions: {len(kg_predictions)} results")
    for entity, score in kg_predictions[:3]:
        print(f"  Entity {entity}: {score:.4f}")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 15 + "QUANTUM NEURO-SYMBOLIC AI DEMO" + " " * 23 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    # Part 1: Neuro-Symbolic AI
    demo_neuro_symbolic()
    
    # Part 2: Quantum Machine Learning
    demo_quantum_ml()
    
    # Part 3: Quantum Neuro-Symbolic Integration
    demo_quantum_neuro_symbolic()
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 25 + "DEMO COMPLETE" + " " * 30 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    print("\nSummary:")
    print("  ✓ Neuro-Symbolic AI: Differentiable logic, KG-guided learning, CBMs")
    print("  ✓ Quantum ML: Hybrid models, quantum kernels, VQC")
    if QISKIT_AVAILABLE:
        print("  ✓ Quantum Neuro-Symbolic: Quantum logic on KG structures")
    else:
        print("  ⚠ Quantum components not available (install Qiskit)")
    
    print("\nTo install quantum dependencies:")
    print("  pip install --break-system-packages qiskit qiskit-aer")


if __name__ == "__main__":
    main()
