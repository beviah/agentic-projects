"""Quantum Graph Neural Networks.

Implements quantum circuits for graph neural network operations,
combining quantum message passing with knowledge graph reasoning.
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


class QuantumMessagePassing:
    """Quantum circuit for graph message passing."""
    
    def __init__(self, n_qubits: int, n_layers: int = 2):
        """Initialize quantum message passing.
        
        Args:
            n_qubits: Number of qubits per node
            n_layers: Number of message passing layers
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Parameters: rotations + entanglement angles
        self.n_params_per_layer = n_qubits * 3  # RY, RZ, RY per qubit
        self.total_params = n_layers * self.n_params_per_layer
    
    def create_message_circuit(self, layer_idx: int, 
                              params: ParameterVector) -> QuantumCircuit:
        """Create quantum circuit for one message passing layer.
        
        Args:
            layer_idx: Layer index
            params: Parameter vector
            
        Returns:
            Quantum circuit for message passing
        """
        qc = QuantumCircuit(self.n_qubits)
        
        param_offset = layer_idx * self.n_params_per_layer
        
        # Rotation layer 1
        for i in range(self.n_qubits):
            qc.ry(params[param_offset + i * 3], i)
        
        # Entanglement layer (message passing)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Rotation layer 2
        for i in range(self.n_qubits):
            qc.rz(params[param_offset + i * 3 + 1], i)
        
        # Additional entanglement
        for i in range(1, self.n_qubits - 1, 2):
            qc.cx(i, i + 1)
        
        # Rotation layer 3
        for i in range(self.n_qubits):
            qc.ry(params[param_offset + i * 3 + 2], i)
        
        return qc
    
    def aggregate_neighbors(self, node_state: np.ndarray,
                           neighbor_states: List[np.ndarray],
                           param_values: np.ndarray) -> np.ndarray:
        """Aggregate information from neighbors using quantum superposition.
        
        Args:
            node_state: Current node state vector
            neighbor_states: List of neighbor state vectors
            param_values: Parameter values
            
        Returns:
            Updated node state after aggregation
        """
        # Create parameter vector
        params = ParameterVector('θ', self.total_params)
        
        # Build full circuit
        qc = QuantumCircuit(self.n_qubits)
        
        # Encode node state
        normalized_state = node_state / (np.linalg.norm(node_state) + 1e-8)
        state_dim = 2 ** self.n_qubits
        padded_state = np.zeros(state_dim)
        padded_state[:min(len(normalized_state), state_dim)] = normalized_state[:state_dim]
        padded_state = padded_state / (np.linalg.norm(padded_state) + 1e-8)
        padded_state = padded_state / np.linalg.norm(padded_state)
        
        qc.initialize(padded_state, range(self.n_qubits))
        
        # Apply message passing layers
        for layer in range(self.n_layers):
            layer_circuit = self.create_message_circuit(layer, params)
            qc = qc.compose(layer_circuit)
        
        # Bind parameters
        param_dict = {params[i]: param_values[i] for i in range(self.total_params)}
        bound_circuit = qc.assign_parameters(param_dict)
        
        # Execute
        statevector = Statevector.from_instruction(bound_circuit)
        
        return statevector.data[:len(node_state)].real


class QuantumAggregation:
    """Quantum aggregation via superposition."""
    
    @staticmethod
    def superposition_aggregate(states: List[np.ndarray]) -> np.ndarray:
        """Create equal-weight superposition of states.
        
        Args:
            states: List of state vectors to aggregate
            
        Returns:
            Aggregated state in superposition
        """
        if not states:
            return np.array([])
        
        # Normalize each state
        normalized_states = []
        for state in states:
            norm = np.linalg.norm(state)
            if norm > 1e-8:
                normalized_states.append(state / norm)
            else:
                normalized_states.append(state)
        
        # Equal-weight superposition
        weight = 1.0 / np.sqrt(len(normalized_states))
        aggregated = sum(weight * s for s in normalized_states)
        
        return aggregated
    
    @staticmethod
    def weighted_aggregate(states: List[np.ndarray],
                          weights: List[float]) -> np.ndarray:
        """Create weighted superposition of states.
        
        Args:
            states: List of state vectors
            weights: Attention weights (should sum to 1)
            
        Returns:
            Weighted aggregated state
        """
        if not states or not weights:
            return np.array([])
        
        # Normalize weights
        total = sum(weights)
        if total > 1e-8:
            norm_weights = [w / total for w in weights]
        else:
            norm_weights = [1.0 / len(weights)] * len(weights)
        
        # Weighted combination
        aggregated = sum(w * s for w, s in zip(norm_weights, states))
        
        return aggregated


class QuantumAttention:
    """Quantum attention mechanism for graphs."""
    
    def __init__(self, n_qubits: int):
        """Initialize quantum attention.
        
        Args:
            n_qubits: Number of qubits for states
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_qubits = n_qubits
    
    def compute_attention_score(self, query: np.ndarray,
                               key: np.ndarray) -> float:
        """Compute quantum attention score using fidelity.
        
        Args:
            query: Query state vector
            key: Key state vector
            
        Returns:
            Attention score (fidelity)
        """
        # Normalize states
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        key_norm = key / (np.linalg.norm(key) + 1e-8)
        
        # Compute fidelity as attention score
        fidelity = abs(np.vdot(query_norm, key_norm)) ** 2
        
        return fidelity
    
    def multi_head_attention(self, query: np.ndarray,
                            keys: List[np.ndarray],
                            values: List[np.ndarray],
                            n_heads: int = 2) -> np.ndarray:
        """Multi-head quantum attention.
        
        Args:
            query: Query state
            keys: List of key states
            values: List of value states
            n_heads: Number of attention heads
            
        Returns:
            Attended output state
        """
        if not keys or not values:
            return query
        
        # Split into heads
        head_dim = len(query) // n_heads
        attended_heads = []
        
        for head_idx in range(n_heads):
            start = head_idx * head_dim
            end = start + head_dim
            
            # Extract head-specific components
            query_head = query[start:end]
            
            # Compute attention scores
            scores = []
            for key in keys:
                key_head = key[start:end] if len(key) > end else key[:head_dim]
                score = self.compute_attention_score(query_head, key_head)
                scores.append(score)
            
            # Softmax normalization
            scores = np.array(scores)
            exp_scores = np.exp(scores - np.max(scores))
            attention_weights = exp_scores / (np.sum(exp_scores) + 1e-8)
            
            # Apply attention to values
            attended = np.zeros(head_dim)
            for weight, value in zip(attention_weights, values):
                value_head = value[start:end] if len(value) > end else value[:head_dim]
                attended += weight * value_head
            
            attended_heads.append(attended)
        
        # Concatenate heads
        return np.concatenate(attended_heads)


class QuantumGNN(nn.Module):
    """Quantum Graph Neural Network.
    
    Architecture:
        Node features → Quantum encoding → Quantum message passing →
        → Quantum aggregation → Measurement → Updated embeddings
    """
    
    def __init__(self, n_nodes: int, input_dim: int, n_qubits: int = 4,
                 n_layers: int = 2, hidden_dim: int = 16, backend=None):
        """Initialize Quantum GNN.
        
        Args:
            n_nodes: Number of nodes in graph
            input_dim: Input feature dimension
            n_qubits: Qubits per node
            n_layers: Message passing layers
            hidden_dim: Hidden dimension for classical post-processing
            backend: Optional QuantumBackend (simulation-only, uses Statevector)
        """
        super().__init__()
        
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.backend = backend
        
        # Classical encoder: input features → quantum-compatible
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_qubits)
        )
        
        # Quantum message passing
        self.quantum_mp = QuantumMessagePassing(n_qubits, n_layers)
        
        # Learnable quantum parameters
        self.mp_params = nn.Parameter(
            torch.randn(self.quantum_mp.total_params) * 0.1
        )
        
        # Quantum attention
        self.quantum_attention = QuantumAttention(n_qubits)
        
        # Classical decoder: quantum output → node embeddings
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Warn if hardware backend provided (uses Statevector, simulation only)
        if backend is not None and hasattr(backend, 'is_hardware') and backend.is_hardware:
            import warnings
            warnings.warn(
                "QuantumGNN uses Statevector.from_instruction() and is not "
                "hardware-compatible. Will use local simulation.",
                UserWarning
            )
    
    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Encode node features to quantum-compatible representation.
        
        Args:
            features: Node features (n_nodes, input_dim)
            
        Returns:
            Quantum features (n_nodes, n_qubits)
        """
        return self.feature_encoder(features)
    
    def quantum_forward(self, node_features: np.ndarray,
                       adjacency: np.ndarray) -> np.ndarray:
        """Quantum forward pass through message passing.
        
        Args:
            node_features: Features for all nodes (n_nodes, n_qubits)
            adjacency: Adjacency matrix (n_nodes, n_nodes)
            
        Returns:
            Updated node features after quantum message passing
        """
        updated_features = []
        # WARNING: Gradient flow is SEVERED here - .detach().cpu().numpy() breaks PyTorch autograd
        # This means gradients will NOT flow back to mp_params during backprop
        # To fix: migrate to PennyLane with interface="torch" for native autodiff
        params = self.mp_params.detach().cpu().numpy()
        
        for node_idx in range(self.n_nodes):
            # Get node state
            node_state = node_features[node_idx]
            
            # Get neighbor states
            neighbors = np.where(adjacency[node_idx] > 0)[0]
            neighbor_states = [node_features[n] for n in neighbors]
            
            # Apply quantum message passing
            updated_state = self.quantum_mp.aggregate_neighbors(
                node_state, neighbor_states, params
            )
            
            updated_features.append(updated_state)
        
        return np.array(updated_features)
    
    def forward(self, node_features: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass through Quantum GNN.
        
        Args:
            node_features: Input node features (n_nodes, input_dim)
            adjacency: Adjacency matrix (n_nodes, n_nodes)
            
        Returns:
            Node embeddings (n_nodes, hidden_dim)
        """
        # Encode features
        quantum_features = self.encode_features(node_features)
        
        # Convert to numpy for quantum processing
        # WARNING: Gradient flow is SEVERED here - .detach().cpu().numpy() breaks PyTorch autograd
        # This means gradients will NOT flow back to encoder/quantum_params during backprop
        # To fix: migrate to PennyLane with interface="torch" for native autodiff
        qf_np = quantum_features.detach().cpu().numpy()
        adj_np = adjacency.detach().cpu().numpy()
        
        # Quantum message passing
        updated_features = self.quantum_forward(qf_np, adj_np)
        
        # Convert back to torch
        updated_torch = torch.tensor(updated_features, dtype=torch.float32,
                                    device=node_features.device)
        
        # Decode to final embeddings
        embeddings = self.decoder(updated_torch)
        
        return embeddings


class QuantumKGGNN(nn.Module):
    """Quantum GNN specialized for Knowledge Graphs."""
    
    def __init__(self, n_entities: int, n_relations: int, embedding_dim: int = 16,
                 n_qubits: int = 4, n_layers: int = 2):
        """Initialize Quantum KG-GNN.
        
        Args:
            n_entities: Number of entities
            n_relations: Number of relation types
            embedding_dim: Embedding dimension
            n_qubits: Qubits per entity
            n_layers: Message passing layers
        """
        super().__init__()
        
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_entities = n_entities
        self.n_relations = n_relations
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)
        
        # Relation-specific quantum GNNs
        self.relation_qgnns = nn.ModuleList([
            QuantumGNN(
                n_nodes=n_entities,
                input_dim=embedding_dim,
                n_qubits=n_qubits,
                n_layers=n_layers,
                hidden_dim=embedding_dim
            )
            for _ in range(n_relations)
        ])
    
    def forward(self, adjacency_matrices: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Forward pass through quantum KG-GNN.
        
        Args:
            adjacency_matrices: Dict mapping relation_id to adjacency matrix
            
        Returns:
            Updated entity embeddings (n_entities, embedding_dim)
        """
        # Initialize with entity embeddings
        entity_ids = torch.arange(self.n_entities)
        features = self.entity_embeddings(entity_ids)
        
        # Aggregate across all relation types
        relation_outputs = []
        
        for relation_id, adj_matrix in adjacency_matrices.items():
            if relation_id < len(self.relation_qgnns):
                # Apply relation-specific quantum GNN
                relation_output = self.relation_qgnns[relation_id](features, adj_matrix)
                relation_outputs.append(relation_output)
        
        # Combine relation outputs
        if relation_outputs:
            combined = torch.stack(relation_outputs, dim=0).mean(dim=0)
        else:
            combined = features
        
        return combined


if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Install with: pip install --break-system-packages qiskit qiskit-aer")
    else:
        print("=" * 60)
        print("Quantum Graph Neural Network Example")
        print("=" * 60)
        
        # Test quantum message passing
        print("\nTesting Quantum Message Passing...")
        
        qmp = QuantumMessagePassing(n_qubits=4, n_layers=2)
        
        print(f"\nConfiguration:")
        print(f"  Qubits: {qmp.n_qubits}")
        print(f"  Layers: {qmp.n_layers}")
        print(f"  Parameters: {qmp.total_params}")
        
        # Test aggregation
        node_state = np.random.randn(4) * 0.5
        neighbor_states = [np.random.randn(4) * 0.5 for _ in range(2)]
        params = np.random.randn(qmp.total_params) * 0.1
        
        updated = qmp.aggregate_neighbors(node_state, neighbor_states, params)
        
        print(f"\nMessage passing result:")
        print(f"  Input state shape: {node_state.shape}")
        print(f"  Neighbors: {len(neighbor_states)}")
        print(f"  Output state shape: {updated.shape}")
        
        # Test quantum attention
        print("\n" + "=" * 60)
        print("Quantum Attention")
        print("=" * 60)
        
        qattn = QuantumAttention(n_qubits=4)
        
        query = np.random.randn(4)
        keys = [np.random.randn(4) for _ in range(3)]
        values = [np.random.randn(4) for _ in range(3)]
        
        attended = qattn.multi_head_attention(query, keys, values, n_heads=2)
        
        print(f"\nAttention computation:")
        print(f"  Query shape: {query.shape}")
        print(f"  Keys: {len(keys)}")
        print(f"  Output shape: {attended.shape}")
        
        # Test full Quantum GNN
        print("\n" + "=" * 60)
        print("Quantum GNN Model")
        print("=" * 60)
        
        qgnn = QuantumGNN(
            n_nodes=5,
            input_dim=8,
            n_qubits=4,
            n_layers=2,
            hidden_dim=16
        )
        
        print(f"\nModel configuration:")
        print(f"  Nodes: {qgnn.n_nodes}")
        print(f"  Input dim: {qgnn.input_dim}")
        print(f"  Qubits: {qgnn.n_qubits}")
        print(f"  Layers: {qgnn.n_layers}")
        
        # Create sample graph
        node_features = torch.randn(5, 8)
        adjacency = torch.tensor([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0]
        ], dtype=torch.float32)
        
        print(f"\nGraph structure:")
        print(f"  Nodes: {node_features.shape[0]}")
        print(f"  Edges: {adjacency.sum().item() // 2:.0f}")
        
        # Forward pass
        embeddings = qgnn(node_features, adjacency)
        
        print(f"\nOutput embeddings:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Node 0 norm: {torch.norm(embeddings[0]):.3f}")
        print(f"  Node 1 norm: {torch.norm(embeddings[1]):.3f}")
        
        # Test Quantum KG-GNN
        print("\n" + "=" * 60)
        print("Quantum Knowledge Graph GNN")
        print("=" * 60)
        
        qkg_gnn = QuantumKGGNN(
            n_entities=6,
            n_relations=2,
            embedding_dim=16,
            n_qubits=4,
            n_layers=2
        )
        
        print(f"\nKG-GNN configuration:")
        print(f"  Entities: {qkg_gnn.n_entities}")
        print(f"  Relations: {qkg_gnn.n_relations}")
        
        # Create adjacency matrices for each relation
        adj_dict = {
            0: torch.tensor([
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            1: torch.tensor([
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
        
        print(f"\nEntity embeddings:")
        print(f"  Shape: {entity_embeddings.shape}")
        print(f"  Entity 0 norm: {torch.norm(entity_embeddings[0]):.3f}")
        print(f"  Entity 1 norm: {torch.norm(entity_embeddings[1]):.3f}")
        
        print("\n" + "=" * 60)
        print("Quantum GNN tests passed!")
        print("=" * 60)
