"""Enhanced Quantum Neuro-Symbolic AI Demo.

Demonstrates the complete quantum neuro-symbolic pipeline including:
- Quantum Concept Bottleneck Models
- Quantum Graph Neural Networks
- Full integration with quantum logic and quantum KG embeddings
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
    print("Warning: Qiskit not available. Install with: pip install --break-system-packages qiskit qiskit-aer")

from neuro_symbolic.differentiable_logic import DifferentiableLogicProgram
from neuro_symbolic.knowledge_guided_nn import KnowledgeGraph, KGGuidedGNN

if QISKIT_AVAILABLE:
    from quantum_neuro_symbolic.quantum_cbm import QuantumCBM, HybridQuantumCBM
    from quantum_neuro_symbolic.quantum_gnn import QuantumGNN, QuantumKGGNN
    from quantum_neuro_symbolic.quantum_logic_circuits import QuantumLogicProgram
    from quantum_neuro_symbolic.quantum_kg_embedding import QuantumKGEmbedding


def demo_quantum_cbm():
    """Demonstrate Quantum Concept Bottleneck Model."""
    print("\n" + "=" * 70)
    print("QUANTUM CONCEPT BOTTLENECK MODEL")
    print("=" * 70)
    
    if not QISKIT_AVAILABLE:
        print("Skipped - Qiskit not available")
        return
    
    print("\n1. Pure Quantum CBM")
    print("-" * 70)
    
    concept_names = ['quantum_entangled', 'superposition_active', 
                     'coherent_state', 'quantum_interference']
    
    qcbm = QuantumCBM(
        input_dim=10,
        n_concepts=4,
        n_classes=3,
        n_qubits=8,
        concept_names=concept_names
    )
    
    print(f"Model: {qcbm.n_qubits} qubits, {qcbm.n_concepts} quantum concepts")
    print(f"Quantum parameters: {qcbm.quantum_params.shape[0]}")
    
    # Test prediction with explanation
    x_test = torch.randn(10)
    explanation = qcbm.predict_with_explanation(x_test, threshold=0.3)
    
    print(f"\nPrediction with quantum concept explanation:")
    print(f"  Predicted class: {explanation['predicted_class']}")
    print(f"  Confidence: {explanation['class_probability']:.3f}")
    print(f"  Active quantum concepts: {len(explanation['active_concepts'])}")
    for concept in explanation['active_concepts']:
        print(f"    - {concept['name']}: {concept['probability']:.3f}")
    
    print("\n2. Hybrid Quantum-Classical CBM")
    print("-" * 70)
    
    hybrid_cbm = HybridQuantumCBM(
        input_dim=10,
        n_concepts=8,
        n_classes=3,
        n_quantum_concepts=4,
        n_classical_concepts=4
    )
    
    x_batch = torch.randn(2, 10)
    logits, classical_concepts, quantum_concepts = hybrid_cbm(x_batch)
    
    print(f"Hybrid model results:")
    print(f"  Classical concepts shape: {classical_concepts.shape}")
    print(f"  Quantum concepts shape: {quantum_concepts.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"\n  Sample 1 - Classical concepts: {classical_concepts[0][:2].detach().numpy()}...")
    print(f"  Sample 1 - Quantum concepts: {quantum_concepts[0][:2].detach().numpy()}...")


def demo_quantum_gnn():
    """Demonstrate Quantum Graph Neural Network."""
    print("\n" + "=" * 70)
    print("QUANTUM GRAPH NEURAL NETWORK")
    print("=" * 70)
    
    if not QISKIT_AVAILABLE:
        print("Skipped - Qiskit not available")
        return
    
    print("\n1. Quantum GNN on General Graph")
    print("-" * 70)
    
    qgnn = QuantumGNN(
        n_nodes=5,
        input_dim=8,
        n_qubits=4,
        n_layers=2,
        hidden_dim=16
    )
    
    print(f"Quantum GNN: {qgnn.n_nodes} nodes, {qgnn.n_qubits} qubits/node")
    print(f"Quantum parameters: {qgnn.mp_params.shape[0]}")
    
    # Create sample graph
    node_features = torch.randn(5, 8)
    adjacency = torch.tensor([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=torch.float32)
    
    n_edges = adjacency.sum().item() // 2
    print(f"\nGraph structure: {qgnn.n_nodes} nodes, {n_edges:.0f} edges")
    
    # Forward pass
    embeddings = qgnn(node_features, adjacency)
    
    print(f"\nQuantum message passing complete:")
    print(f"  Output embeddings shape: {embeddings.shape}")
    print(f"  Node embeddings:")
    for i in range(3):
        print(f"    Node {i}: norm = {torch.norm(embeddings[i]):.3f}")
    
    print("\n2. Quantum Knowledge Graph GNN")
    print("-" * 70)
    
    qkg_gnn = QuantumKGGNN(
        n_entities=6,
        n_relations=2,
        embedding_dim=16,
        n_qubits=4,
        n_layers=2
    )
    
    print(f"Quantum KG-GNN: {qkg_gnn.n_entities} entities, {qkg_gnn.n_relations} relations")
    
    # Create adjacency matrices for relations
    adj_dict = {
        0: torch.tensor([  # is-a relation
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]
        ], dtype=torch.float32),
        1: torch.tensor([  # part-of relation
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=torch.float32)
    }
    
    # Forward pass
    entity_embeddings = qkg_gnn(adj_dict)
    
    print(f"\nQuantum KG processing complete:")
    print(f"  Entity embeddings shape: {entity_embeddings.shape}")
    print(f"  Entity embeddings:")
    for i in range(3):
        print(f"    Entity {i}: norm = {torch.norm(entity_embeddings[i]):.3f}")


def demo_full_quantum_pipeline():
    """Demonstrate complete quantum neuro-symbolic pipeline."""
    print("\n" + "=" * 70)
    print("FULL QUANTUM NEURO-SYMBOLIC PIPELINE")
    print("=" * 70)
    
    if not QISKIT_AVAILABLE:
        print("Skipped - Qiskit not available")
        return
    
    print("\nIntegrated System Architecture:")
    print("-" * 70)
    print("  Input Data")
    print("      ↓")
    print("  [1] Quantum Knowledge Graph Embedding")
    print("      ↓")
    print("  [2] Quantum Graph Neural Network")
    print("      ↓")
    print("  [3] Quantum Logic Reasoning")
    print("      ↓")
    print("  [4] Quantum Concept Bottleneck")
    print("      ↓")
    print("  Output Prediction + Explanation")
    
    print("\n" + "-" * 70)
    print("Executing Pipeline...")
    print("-" * 70)
    
    # Step 1: Quantum KG Embedding
    print("\n[1] Quantum Knowledge Graph Embedding")
    qkg = QuantumKGEmbedding(n_entities=5, n_relations=2, n_qubits=4)
    print(f"    Initialized: {qkg.n_entities} entities, {qkg.n_qubits} qubits/entity")
    
    # Score some triples
    triples = [(0, 0, 1), (1, 0, 2), (2, 1, 3)]
    print(f"    Triple scoring:")
    for h, r, t in triples:
        score = qkg.compute_triple_score(h, r, t)
        print(f"      ({h}, {r}, {t}): {score:.4f}")
    
    # Step 2: Quantum GNN
    print("\n[2] Quantum Graph Neural Network")
    qgnn = QuantumKGGNN(
        n_entities=5,
        n_relations=2,
        embedding_dim=16,
        n_qubits=4,
        n_layers=2
    )
    
    adj_dict = {
        0: torch.eye(5),
        1: torch.tensor([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype=torch.float32)
    }
    
    entity_embeddings = qgnn(adj_dict)
    print(f"    Generated entity embeddings: {entity_embeddings.shape}")
    print(f"    Average embedding norm: {torch.norm(entity_embeddings, dim=1).mean():.3f}")
    
    # Step 3: Quantum Logic Reasoning
    print("\n[3] Quantum Logic Reasoning")
    rules = [
        (3, [0], 'implication'),
        (4, [1], 'implication'),
    ]
    qlogic = QuantumLogicProgram(n_predicates=5, rules=rules)
    
    initial_facts = np.array([0.8, 0.6, 0.0, 0.0, 0.0])
    params = np.random.randn(qlogic.n_params) * 0.1
    
    derived_facts = qlogic.forward(initial_facts, params)
    
    print(f"    Initial facts: {initial_facts[:3]}")
    print(f"    Derived facts: {derived_facts[:3]}")
    print(f"    Logic inference complete")
    
    # Step 4: Quantum CBM
    print("\n[4] Quantum Concept Bottleneck Model")
    concept_names = ['concept_a', 'concept_b', 'concept_c']
    qcbm = QuantumCBM(
        input_dim=16,
        n_concepts=3,
        n_classes=5,
        n_qubits=8,
        concept_names=concept_names
    )
    
    # Use entity embeddings as input
    sample_input = entity_embeddings[0]
    explanation = qcbm.predict_with_explanation(sample_input, threshold=0.2)
    
    print(f"    Final prediction: class {explanation['predicted_class']}")
    print(f"    Confidence: {explanation['class_probability']:.3f}")
    print(f"    Active concepts: {len(explanation['active_concepts'])}")
    for concept in explanation['active_concepts']:
        print(f"      - {concept['name']}: {concept['probability']:.3f}")
    
    print("\n" + "-" * 70)
    print("✓ Pipeline execution complete!")
    print("-" * 70)
    
    print("\nKey Achievements:")
    print("  ✓ Quantum KG embeddings with exponential state space")
    print("  ✓ Quantum message passing on graph structure")
    print("  ✓ Quantum logic reasoning with differentiable parameters")
    print("  ✓ Interpretable quantum concepts")
    print("  ✓ End-to-end quantum neuro-symbolic inference")


def demo_comparison():
    """Compare classical vs quantum components."""
    print("\n" + "=" * 70)
    print("CLASSICAL VS QUANTUM COMPARISON")
    print("=" * 70)
    
    if not QISKIT_AVAILABLE:
        print("Skipped - Qiskit not available")
        return
    
    print("\n1. Logic Reasoning Comparison")
    print("-" * 70)
    
    rules = [(2, [0]), (2, [1])]
    initial = np.array([0.9, 0.1, 0.0])
    
    # Classical
    classical_logic = DifferentiableLogicProgram(n_predicates=3, rules=rules)
    classical_result = classical_logic(torch.tensor([initial], dtype=torch.float32))
    classical_result = classical_result[0].detach().numpy()
    
    # Quantum
    quantum_rules = [(h, b, 'implication') for h, b in rules]
    quantum_logic = QuantumLogicProgram(n_predicates=3, rules=quantum_rules)
    quantum_params = np.random.randn(quantum_logic.n_params) * 0.1
    quantum_result = quantum_logic.forward(initial, quantum_params)
    
    print(f"Initial facts: {initial}")
    print(f"Classical result: {classical_result}")
    print(f"Quantum result: {quantum_result}")
    print(f"Difference: {np.abs(classical_result - quantum_result).mean():.4f}")
    
    print("\n2. GNN Comparison")
    print("-" * 70)
    
    kg = KnowledgeGraph(n_entities=5, n_relations=2)
    kg.add_triple(0, 0, 1)
    kg.add_triple(1, 0, 2)
    kg.add_triple(2, 1, 3)
    
    # Classical GNN
    classical_gnn = KGGuidedGNN(n_entities=5, n_relations=2, 
                                embedding_dim=16, hidden_dim=16, n_layers=2)
    classical_embeddings = classical_gnn(kg)
    
    # Quantum GNN
    quantum_gnn = QuantumKGGNN(n_entities=5, n_relations=2,
                              embedding_dim=16, n_qubits=4, n_layers=2)
    adj_dict = {0: torch.eye(5), 1: kg.to_adjacency_matrix(1)}
    quantum_embeddings = quantum_gnn(adj_dict)
    
    print(f"Classical GNN output shape: {classical_embeddings.shape}")
    print(f"Quantum GNN output shape: {quantum_embeddings.shape}")
    print(f"Classical avg norm: {torch.norm(classical_embeddings, dim=1).mean():.3f}")
    print(f"Quantum avg norm: {torch.norm(quantum_embeddings, dim=1).mean():.3f}")
    
    print("\nNote: Quantum components offer:")
    print("  • Exponential state space (2^n with n qubits)")
    print("  • Quantum superposition for parallel processing")
    print("  • Entanglement for complex correlations")
    print("  • Potential quantum advantage for specific tasks")


def main():
    """Run enhanced quantum neuro-symbolic demonstrations."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 10 + "ENHANCED QUANTUM NEURO-SYMBOLIC AI DEMO" + " " * 18 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    print("\nThis demo showcases:")
    print("  • Quantum Concept Bottleneck Models (NEW)")
    print("  • Quantum Graph Neural Networks (NEW)")
    print("  • Integrated quantum neuro-symbolic pipeline")
    print("  • Classical vs quantum comparisons")
    
    # Demonstrations
    demo_quantum_cbm()
    demo_quantum_gnn()
    demo_full_quantum_pipeline()
    demo_comparison()
    
    # Summary
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 25 + "DEMO COMPLETE" + " " * 30 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    print("\n📊 Components Demonstrated:")
    if QISKIT_AVAILABLE:
        print("  ✓ Quantum Concept Bottleneck Models")
        print("  ✓ Quantum Graph Neural Networks")
        print("  ✓ Quantum Logic Circuits")
        print("  ✓ Quantum Knowledge Graph Embeddings")
        print("  ✓ Full quantum neuro-symbolic pipeline")
        print("  ✓ Classical-quantum comparisons")
    else:
        print("  ⚠ Quantum components require Qiskit")
        print("\n  Install with:")
        print("    pip install --break-system-packages qiskit qiskit-aer")
    
    print("\n🎯 Research Contributions:")
    print("  • Novel quantum concept extraction")
    print("  • Quantum message passing on graphs")
    print("  • Interpretable quantum reasoning")
    print("  • Hybrid quantum-classical architectures")
    
    print("\n📚 Next Steps:")
    print("  1. Run benchmarks: python benchmarks/run_benchmarks.py")
    print("  2. Add test suite: python -m pytest tests/")
    print("  3. Analyze quantum advantage")
    print("  4. Deploy on real quantum hardware")


if __name__ == "__main__":
    main()
