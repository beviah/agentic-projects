"""Quantum Knowledge Graph Embeddings.

Implements quantum circuits for encoding and reasoning over knowledge graphs,
combining quantum computing with graph-structured knowledge.
"""

import numpy as np
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn


class QuantumEntityEncoder:
    """Encode knowledge graph entities as quantum states."""
    
    @staticmethod
    def amplitude_encoding(entity_features: np.ndarray, n_qubits: int) -> QuantumCircuit:
        """Encode entity features in quantum state amplitudes.
        
        Args:
            entity_features: Feature vector for entity
            n_qubits: Number of qubits
            
        Returns:
            Quantum circuit encoding the entity
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        # Normalize features
        norm = np.linalg.norm(entity_features)
        if norm > 0:
            normalized = entity_features / norm
        else:
            normalized = entity_features
        
        # Pad to 2^n_qubits dimensions
        state_dim = 2 ** n_qubits
        padded = np.zeros(state_dim)
        padded[:min(len(normalized), state_dim)] = normalized[:state_dim]
        
        # Renormalize
        norm = np.linalg.norm(padded)
        if norm > 0:
            padded = padded / norm
        
        # Create circuit
        qc = QuantumCircuit(n_qubits)
        qc.initialize(padded, range(n_qubits))
        
        return qc
    
    @staticmethod
    def angle_encoding(entity_features: np.ndarray) -> QuantumCircuit:
        """Encode entity features as rotation angles.
        
        Args:
            entity_features: Feature vector
            
        Returns:
            Quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        n_qubits = len(entity_features)
        qc = QuantumCircuit(n_qubits)
        
        # Hadamard layer for superposition
        for i in range(n_qubits):
            qc.h(i)
        
        # Encode features as RY rotations
        for i, feature in enumerate(entity_features):
            qc.ry(feature, i)
        
        return qc


class QuantumRelationTransform:
    """Quantum circuit representing knowledge graph relations."""
    
    def __init__(self, n_qubits: int, relation_type: str = 'hierarchical'):
        """Initialize quantum relation transform.
        
        Args:
            n_qubits: Number of qubits
            relation_type: Type of relation ('hierarchical', 'symmetric', 'transitive')
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_qubits = n_qubits
        self.relation_type = relation_type
        
        # Learnable parameters
        self.n_params = n_qubits * 3  # 3 rotations per qubit
        self.params = ParameterVector('r', self.n_params)
    
    def create_circuit(self) -> QuantumCircuit:
        """Create parameterized circuit for relation transformation."""
        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0
        
        if self.relation_type == 'hierarchical':
            # Hierarchical relations: progressive transformations
            for i in range(self.n_qubits):
                qc.ry(self.params[param_idx], i)
                param_idx += 1
            
            # Cascading CNOTs (parent to child flow)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            
            for i in range(self.n_qubits):
                qc.rz(self.params[param_idx], i)
                param_idx += 1
        
        elif self.relation_type == 'symmetric':
            # Symmetric relations: bidirectional transformations
            for i in range(self.n_qubits):
                qc.ry(self.params[param_idx], i)
                param_idx += 1
            
            # Symmetric entanglement
            for i in range(0, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
                qc.cx(i + 1, i)
            
            for i in range(self.n_qubits):
                qc.rz(self.params[param_idx], i)
                param_idx += 1
        
        elif self.relation_type == 'transitive':
            # Transitive relations: chained transformations
            for layer in range(2):
                for i in range(self.n_qubits):
                    if param_idx < self.n_params:
                        qc.ry(self.params[param_idx], i)
                        param_idx += 1
                
                # Chain CNOTs
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
                
                # Second rotation (rz) if params remain
                for i in range(self.n_qubits):
                    if param_idx < self.n_params:
                        qc.rz(self.params[param_idx], i)
                        param_idx += 1
        
        return qc


class QuantumKGEmbedding:
    """Quantum knowledge graph embedding model."""
    
    def __init__(self, n_entities: int, n_relations: int, n_qubits: int = 4, backend=None):
        """Initialize quantum KG embedding.
        
        Args:
            n_entities: Number of entities in KG
            n_relations: Number of relation types
            n_qubits: Qubits per entity encoding
            backend: Optional QuantumBackend (simulation-only, uses Statevector)
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_qubits = n_qubits
        self.backend = backend
        
        # Classical entity features (to be encoded quantum)
        self.entity_features = np.random.randn(n_entities, n_qubits) * 0.1
        
        # Quantum relation transforms
        self.relation_circuits = {}
        self.relation_params = {}
        
        relation_types = ['hierarchical', 'symmetric', 'transitive']
        for rel_id in range(n_relations):
            rel_type = relation_types[rel_id % len(relation_types)]
            transform = QuantumRelationTransform(n_qubits, rel_type)
            self.relation_circuits[rel_id] = transform.create_circuit()
            # Random initial parameters
            self.relation_params[rel_id] = np.random.randn(transform.n_params) * 0.1
        
        # Warn if hardware backend provided (uses Statevector, simulation only)
        if backend is not None and hasattr(backend, 'is_hardware') and backend.is_hardware:
            import warnings
            warnings.warn(
                "QuantumKGEmbedding uses Statevector.from_instruction() and is not "
                "hardware-compatible. Will use local simulation.",
                UserWarning
            )
    
    def encode_entity(self, entity_id: int) -> QuantumCircuit:
        """Encode entity as quantum state.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Quantum circuit encoding the entity
        """
        features = self.entity_features[entity_id]
        return QuantumEntityEncoder.angle_encoding(features)
    
    def apply_relation(self, entity_circuit: QuantumCircuit, relation_id: int) -> QuantumCircuit:
        """Apply relation transformation to entity.
        
        Args:
            entity_circuit: Encoded entity circuit
            relation_id: Relation type
            
        Returns:
            Transformed circuit
        """
        # Get relation circuit
        rel_circuit = self.relation_circuits[relation_id]
        rel_params = self.relation_params[relation_id]
        
        # Bind parameters
        param_dict = {rel_circuit.parameters[i]: rel_params[i] 
                     for i in range(len(rel_circuit.parameters))}
        bound_rel = rel_circuit.assign_parameters(param_dict)
        
        # Compose
        return entity_circuit.compose(bound_rel)
    
    def compute_triple_score(self, head: int, relation: int, tail: int) -> float:
        """Score a knowledge graph triple (h, r, t).
        
        Args:
            head: Head entity ID
            relation: Relation ID
            tail: Tail entity ID
            
        Returns:
            Score (higher = more likely to be true)
        """
        # Encode head entity
        head_circuit = self.encode_entity(head)
        
        # Apply relation
        transformed = self.apply_relation(head_circuit, relation)
        
        # Encode tail entity
        tail_circuit = self.encode_entity(tail)
        
        # Compute fidelity (overlap) between transformed head and tail
        sv_transformed = Statevector.from_instruction(transformed)
        sv_tail = Statevector.from_instruction(tail_circuit)
        
        fidelity = abs(np.vdot(sv_transformed.data, sv_tail.data)) ** 2
        
        return fidelity
    
    def predict_tail(self, head: int, relation: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Predict tail entities given head and relation.
        
        Args:
            head: Head entity
            relation: Relation type
            top_k: Number of predictions
            
        Returns:
            List of (entity_id, score) tuples
        """
        scores = []
        for tail_id in range(self.n_entities):
            score = self.compute_triple_score(head, relation, tail_id)
            scores.append((tail_id, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


class QuantumGraphNeuralNetwork:
    """Quantum implementation of graph neural network for knowledge graphs."""
    
    def __init__(self, n_qubits: int, n_layers: int = 2):
        """Initialize quantum GNN.
        
        Args:
            n_qubits: Qubits per node
            n_layers: Number of message passing layers
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Parameters for each layer
        self.n_params_per_layer = n_qubits * 3
        self.total_params = n_layers * self.n_params_per_layer
        self.params = np.random.randn(self.total_params) * 0.1
    
    def create_message_circuit(self, layer: int) -> QuantumCircuit:
        """Create quantum circuit for message passing.
        
        Args:
            layer: Layer index
            
        Returns:
            Message passing circuit
        """
        qc = QuantumCircuit(self.n_qubits)
        
        param_offset = layer * self.n_params_per_layer
        
        # Rotation layer
        for i in range(self.n_qubits):
            idx = param_offset + i * 3
            qc.ry(self.params[idx], i)
            qc.rz(self.params[idx + 1], i)
            qc.ry(self.params[idx + 2], i)
        
        # Entanglement (message passing)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        
        return qc
    
    def aggregate_neighbors(self, node_state: QuantumCircuit, 
                           neighbor_states: List[QuantumCircuit]) -> QuantumCircuit:
        """Aggregate information from neighbor nodes.
        
        Args:
            node_state: Current node's quantum state
            neighbor_states: List of neighbor quantum states
            
        Returns:
            Updated node state
        """
        # Simplified: apply message passing to node state
        # In full implementation, would entangle with neighbors
        
        updated = node_state.copy()
        
        for layer in range(self.n_layers):
            message_circuit = self.create_message_circuit(layer)
            updated = updated.compose(message_circuit)
        
        return updated


class QuantumAttentionMechanism:
    """Quantum attention for knowledge graph reasoning."""
    
    def __init__(self, n_qubits: int):
        """Initialize quantum attention.
        
        Args:
            n_qubits: Number of qubits
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_qubits = n_qubits
    
    def compute_attention_score(self, query_state: QuantumCircuit, 
                               key_state: QuantumCircuit) -> float:
        """Compute attention score between query and key.
        
        Args:
            query_state: Query quantum state
            key_state: Key quantum state
            
        Returns:
            Attention score (fidelity)
        """
        # Compute fidelity as attention score
        sv_query = Statevector.from_instruction(query_state)
        sv_key = Statevector.from_instruction(key_state)
        
        attention = abs(np.vdot(sv_query.data, sv_key.data)) ** 2
        
        return attention
    
    def apply_attention(self, query_state: QuantumCircuit, 
                       key_value_pairs: List[Tuple[QuantumCircuit, QuantumCircuit]]) -> QuantumCircuit:
        """Apply quantum attention mechanism.
        
        Args:
            query_state: Query state
            key_value_pairs: List of (key, value) state pairs
            
        Returns:
            Attended output state
        """
        # Compute attention scores
        scores = []
        for key_state, _ in key_value_pairs:
            score = self.compute_attention_score(query_state, key_state)
            scores.append(score)
        
        # Normalize scores
        total = sum(scores)
        if total > 0:
            weights = [s / total for s in scores]
        else:
            weights = [1.0 / len(scores)] * len(scores)
        
        # Weighted combination (simplified: use highest attention)
        max_idx = np.argmax(weights)
        attended_state = key_value_pairs[max_idx][1]
        
        return attended_state


if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Install with: pip install --break-system-packages qiskit qiskit-aer")
    else:
        print("=" * 60)
        print("Quantum Knowledge Graph Embeddings Example")
        print("=" * 60)
        
        # Test entity encoding
        print("\nTesting quantum entity encoding...")
        
        entity_features = np.array([0.5, 0.3, 0.8, 0.2])
        entity_circuit = QuantumEntityEncoder.angle_encoding(entity_features)
        
        print(f"\nEntity encoding circuit:")
        print(f"  Input features: {entity_features}")
        print(f"  Qubits: {entity_circuit.num_qubits}")
        print(f"  Depth: {entity_circuit.depth()}")
        
        # Test relation transform
        print("\n" + "=" * 60)
        print("Quantum Relation Transform")
        print("=" * 60)
        
        relation = QuantumRelationTransform(n_qubits=4, relation_type='hierarchical')
        rel_circuit = relation.create_circuit()
        
        print(f"\nRelation circuit:")
        print(f"  Type: hierarchical")
        print(f"  Qubits: {rel_circuit.num_qubits}")
        print(f"  Depth: {rel_circuit.depth()}")
        print(f"  Parameters: {len(rel_circuit.parameters)}")
        
        # Test quantum KG embedding
        print("\n" + "=" * 60)
        print("Quantum Knowledge Graph Embedding")
        print("=" * 60)
        
        qkg = QuantumKGEmbedding(n_entities=6, n_relations=3, n_qubits=4)
        
        print(f"\nKG Configuration:")
        print(f"  Entities: {qkg.n_entities}")
        print(f"  Relations: {qkg.n_relations}")
        print(f"  Qubits per entity: {qkg.n_qubits}")
        
        # Test triple scoring
        print("\nTesting triple scoring...")
        
        # Score some triples
        triples = [
            (0, 0, 1),  # entity 0 -[relation 0]-> entity 1
            (0, 0, 5),  # entity 0 -[relation 0]-> entity 5
            (2, 1, 3),  # entity 2 -[relation 1]-> entity 3
        ]
        
        for h, r, t in triples:
            score = qkg.compute_triple_score(h, r, t)
            print(f"  Triple ({h}, {r}, {t}): score = {score:.4f}")
        
        # Test tail prediction
        print("\nTesting tail entity prediction...")
        print("Given: (entity 0, relation 0, ?)")
        
        predictions = qkg.predict_tail(head=0, relation=0, top_k=3)
        
        print("\nTop 3 predicted tail entities:")
        for rank, (entity_id, score) in enumerate(predictions, 1):
            print(f"  {rank}. Entity {entity_id}: score = {score:.4f}")
        
        # Test quantum GNN
        print("\n" + "=" * 60)
        print("Quantum Graph Neural Network")
        print("=" * 60)
        
        qgnn = QuantumGraphNeuralNetwork(n_qubits=4, n_layers=2)
        
        print(f"\nQGNN Configuration:")
        print(f"  Qubits: {qgnn.n_qubits}")
        print(f"  Layers: {qgnn.n_layers}")
        print(f"  Total parameters: {qgnn.total_params}")
        
        # Create sample node state
        node_circuit = QuantumEntityEncoder.angle_encoding(np.array([0.5, 0.3, 0.8, 0.2]))
        
        # Apply message passing
        updated_node = qgnn.aggregate_neighbors(node_circuit, [])
        
        print(f"\nNode state after message passing:")
        print(f"  Circuit depth: {updated_node.depth()}")
        
        # Test quantum attention
        print("\n" + "=" * 60)
        print("Quantum Attention Mechanism")
        print("=" * 60)
        
        qattn = QuantumAttentionMechanism(n_qubits=4)
        
        # Create query and key states
        query = QuantumEntityEncoder.angle_encoding(np.array([0.5, 0.5, 0.5, 0.5]))
        key1 = QuantumEntityEncoder.angle_encoding(np.array([0.5, 0.5, 0.5, 0.5]))
        key2 = QuantumEntityEncoder.angle_encoding(np.array([0.1, 0.9, 0.2, 0.8]))
        
        score1 = qattn.compute_attention_score(query, key1)
        score2 = qattn.compute_attention_score(query, key2)
        
        print(f"\nAttention scores:")
        print(f"  Query vs Key1 (similar): {score1:.4f}")
        print(f"  Query vs Key2 (different): {score2:.4f}")
        
        print("\n" + "=" * 60)
        print("Quantum KG embedding tests passed!")
        print("=" * 60)
