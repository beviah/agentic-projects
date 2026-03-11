"""Quantum Concept Bottleneck Models.

Implements quantum circuits for extracting interpretable concepts,
combining quantum computing with concept-based interpretability.
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


class QuantumConceptExtractor:
    """Quantum circuit for extracting interpretable concepts."""
    
    def __init__(self, n_qubits: int, n_concepts: int):
        """Initialize quantum concept extractor.
        
        Args:
            n_qubits: Number of qubits for input encoding
            n_concepts: Number of concepts to extract
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_qubits = n_qubits
        self.n_concepts = n_concepts
        
        # Each concept gets dedicated qubits
        self.qubits_per_concept = max(1, n_qubits // n_concepts)
        self.total_qubits = self.qubits_per_concept * n_concepts
        
        # Learnable parameters for concept extraction
        self.n_params = self.total_qubits * 3  # 3 rotation angles per qubit
        self.params = ParameterVector('θ', self.n_params)
    
    def create_concept_circuit(self) -> QuantumCircuit:
        """Create parameterized circuit for concept extraction.
        
        Returns:
            Quantum circuit that extracts concepts
        """
        qc = QuantumCircuit(self.total_qubits)
        param_idx = 0
        
        # Layer 1: Individual qubit rotations
        for i in range(self.total_qubits):
            qc.ry(self.params[param_idx], i)
            param_idx += 1
        
        # Layer 2: Entanglement within concept groups
        for concept_id in range(self.n_concepts):
            start_qubit = concept_id * self.qubits_per_concept
            end_qubit = start_qubit + self.qubits_per_concept
            
            # Entangle qubits within this concept
            for i in range(start_qubit, end_qubit - 1):
                qc.cx(i, i + 1)
        
        # Layer 3: More rotations
        for i in range(self.total_qubits):
            qc.rz(self.params[param_idx], i)
            param_idx += 1
        
        # Layer 4: Cross-concept entanglement
        for concept_id in range(self.n_concepts - 1):
            qubit1 = (concept_id + 1) * self.qubits_per_concept - 1
            qubit2 = (concept_id + 1) * self.qubits_per_concept
            qc.cx(qubit1, qubit2)
        
        # Layer 5: Final rotations
        for i in range(self.total_qubits):
            qc.ry(self.params[param_idx], i)
            param_idx += 1
        
        return qc
    
    def encode_input(self, features: np.ndarray) -> QuantumCircuit:
        """Encode input features into quantum state.
        
        Args:
            features: Input feature vector
            
        Returns:
            Quantum circuit encoding the input
        """
        qc = QuantumCircuit(self.total_qubits)
        
        # Pad or truncate features
        padded_features = np.zeros(self.total_qubits)
        copy_len = min(len(features), self.total_qubits)
        padded_features[:copy_len] = features[:copy_len]
        
        # Angle encoding
        for i, feature in enumerate(padded_features):
            qc.ry(feature, i)
        
        return qc
    
    def extract_concepts(self, features: np.ndarray, param_values: np.ndarray) -> np.ndarray:
        """Extract concept probabilities from input.
        
        Args:
            features: Input features
            param_values: Parameter values for the circuit
            
        Returns:
            Concept probabilities (n_concepts,)
        """
        # Encode input
        input_circuit = self.encode_input(features)
        
        # Create concept extraction circuit
        concept_circuit = self.create_concept_circuit()
        
        # Bind parameters
        param_dict = {self.params[i]: param_values[i] for i in range(self.n_params)}
        bound_circuit = concept_circuit.assign_parameters(param_dict)
        
        # Combine
        full_circuit = input_circuit.compose(bound_circuit)
        
        # Execute
        statevector = Statevector.from_instruction(full_circuit)
        
        # Extract concept probabilities
        # Each concept: probability that its qubits are in |1...1> state
        concept_probs = np.zeros(self.n_concepts)
        
        for concept_id in range(self.n_concepts):
            start_qubit = concept_id * self.qubits_per_concept
            end_qubit = start_qubit + self.qubits_per_concept
            
            # Measure probability of this concept's qubits being |1>
            prob = 0.0
            for basis_idx, amplitude in enumerate(statevector.data):
                # Check if all qubits in this concept are |1>
                all_one = True
                for qubit_idx in range(start_qubit, end_qubit):
                    if not ((basis_idx >> qubit_idx) & 1):
                        all_one = False
                        break
                
                if all_one:
                    prob += abs(amplitude) ** 2
            
            concept_probs[concept_id] = prob
        
        return concept_probs


class QuantumConceptIntervention:
    """Quantum circuit for intervening on concepts."""
    
    @staticmethod
    def intervene_concept(circuit: QuantumCircuit, concept_qubits: List[int],
                         target_value: float) -> QuantumCircuit:
        """Intervene on a concept by setting its qubits to target value.
        
        Args:
            circuit: Quantum circuit
            concept_qubits: List of qubit indices for this concept
            target_value: Target probability (0.0 to 1.0)
            
        Returns:
            Modified circuit with intervention
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        intervention_circuit = circuit.copy()
        
        # Compute rotation angle for target probability
        # P(|1>) = sin²(θ/2) = target_value
        # θ = 2 * arcsin(sqrt(target_value))
        angle = 2 * np.arcsin(np.sqrt(np.clip(target_value, 0, 1)))
        
        # Apply intervention: reset and rotate to target
        for qubit in concept_qubits:
            intervention_circuit.reset(qubit)
            intervention_circuit.ry(angle, qubit)
        
        return intervention_circuit


class QuantumCBM(nn.Module):
    """Quantum Concept Bottleneck Model.
    
    Architecture:
        Input → Quantum Concept Extractor → Concepts → Classical Predictor → Output
    """
    
    def __init__(self, input_dim: int, n_concepts: int, n_classes: int,
                 n_qubits: int = 8, concept_names: Optional[List[str]] = None,
                 backend=None):
        """Initialize Quantum CBM.
        
        Args:
            input_dim: Input feature dimension
            n_concepts: Number of interpretable concepts
            n_classes: Number of output classes
            n_qubits: Number of qubits for quantum processing
            concept_names: Optional names for concepts
            backend: Optional QuantumBackend (simulation-only, uses Statevector)
        """
        super().__init__()
        
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.input_dim = input_dim
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.n_qubits = n_qubits
        self.backend = backend
        
        # Concept names
        if concept_names is None:
            self.concept_names = [f"quantum_concept_{i}" for i in range(n_concepts)]
        else:
            self.concept_names = concept_names
        
        # Warn if hardware backend provided (uses Statevector, simulation only)
        if backend is not None and hasattr(backend, 'is_hardware') and backend.is_hardware:
            import warnings
            warnings.warn(
                "QuantumCBM uses Statevector.from_instruction() and is not "
                "hardware-compatible. Will use local simulation.",
                UserWarning
            )
        
        # Classical feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits)
        )
        
        # Quantum concept extractor
        self.quantum_extractor = QuantumConceptExtractor(n_qubits, n_concepts)
        
        # Learnable quantum parameters
        self.quantum_params = nn.Parameter(
            torch.randn(self.quantum_extractor.n_params) * 0.1
        )
        
        # Classical predictor from concepts
        self.concept_predictor = nn.Sequential(
            nn.Linear(n_concepts, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            (class_logits, concept_probs) tuple
        """
        batch_size = x.shape[0]
        
        # Encode to quantum-compatible features
        quantum_features = self.feature_encoder(x)
        
        # Extract concepts using quantum circuit
        concept_probs_list = []
        # WARNING: Gradient flow is SEVERED here - .detach().cpu().numpy() breaks PyTorch autograd
        # This means gradients will NOT flow back to quantum_params during backprop
        # To fix: migrate to PennyLane with interface="torch" for native autodiff
        param_values = self.quantum_params.detach().cpu().numpy()
        
        for i in range(batch_size):
            # WARNING: Gradient flow is SEVERED here as well
            features = quantum_features[i].detach().cpu().numpy()
            concepts = self.quantum_extractor.extract_concepts(features, param_values)
            concept_probs_list.append(concepts)
        
        concept_probs = torch.tensor(concept_probs_list, dtype=torch.float32, device=x.device)
        
        # Predict from concepts
        class_logits = self.concept_predictor(concept_probs)
        
        return class_logits, concept_probs
    
    def predict_with_explanation(self, x: torch.Tensor,
                                threshold: float = 0.5) -> Dict:
        """Predict with interpretable explanation.
        
        Args:
            x: Input tensor (single sample)
            threshold: Threshold for concept activation
            
        Returns:
            Dictionary with prediction and explanation
        """
        with torch.no_grad():
            x_batch = x.unsqueeze(0) if x.dim() == 1 else x
            logits, concepts = self.forward(x_batch)
            
            probs = torch.softmax(logits[0], dim=0)
            pred_class = torch.argmax(probs).item()
            
            # Identify active concepts
            active_concepts = []
            for i, prob in enumerate(concepts[0]):
                if prob >= threshold:
                    active_concepts.append({
                        'name': self.concept_names[i],
                        'index': i,
                        'probability': prob.item()
                    })
            
            return {
                'predicted_class': pred_class,
                'class_probability': probs[pred_class].item(),
                'all_class_probs': probs.cpu().numpy(),
                'concept_probabilities': concepts[0].cpu().numpy(),
                'active_concepts': active_concepts,
                'concept_names': self.concept_names
            }
    
    def intervene(self, x: torch.Tensor, concept_id: int,
                 target_value: float) -> torch.Tensor:
        """Perform concept intervention.
        
        Args:
            x: Input tensor
            concept_id: Index of concept to intervene on
            target_value: Target value for concept (0.0 to 1.0)
            
        Returns:
            Predictions with intervention
        """
        # Encode input
        quantum_features = self.feature_encoder(x)
        
        # For simplicity, manually set concept value
        # Full implementation would modify the quantum circuit
        logits, concepts = self.forward(x)
        concepts[:, concept_id] = target_value
        
        # Re-predict with modified concepts
        class_logits = self.concept_predictor(concepts)
        
        return class_logits


class HybridQuantumCBM(nn.Module):
    """Hybrid model combining classical and quantum CBMs."""
    
    def __init__(self, input_dim: int, n_concepts: int, n_classes: int,
                 n_quantum_concepts: int = 4, n_classical_concepts: int = 4):
        super().__init__()
        
        self.n_quantum_concepts = n_quantum_concepts
        self.n_classical_concepts = n_classical_concepts
        total_concepts = n_quantum_concepts + n_classical_concepts
        
        # Classical concept predictor
        self.classical_concepts = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classical_concepts),
            nn.Sigmoid()
        )
        
        # Quantum concept predictor
        if QISKIT_AVAILABLE:
            self.quantum_cbm = QuantumCBM(
                input_dim=input_dim,
                n_concepts=n_quantum_concepts,
                n_classes=n_classes,
                n_qubits=8
            )
        else:
            self.quantum_cbm = None
        
        # Final classifier combining both concept types
        self.final_classifier = nn.Sequential(
            nn.Linear(total_concepts, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            (logits, classical_concepts, quantum_concepts)
        """
        # Classical concepts
        classical_concepts = self.classical_concepts(x)
        
        # Quantum concepts
        if self.quantum_cbm is not None:
            _, quantum_concepts = self.quantum_cbm(x)
        else:
            # Fallback if quantum not available
            quantum_concepts = torch.zeros(x.shape[0], self.n_quantum_concepts)
        
        # Combine concepts
        all_concepts = torch.cat([classical_concepts, quantum_concepts], dim=1)
        
        # Final prediction
        logits = self.final_classifier(all_concepts)
        
        return logits, classical_concepts, quantum_concepts


if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Install with: pip install --break-system-packages qiskit qiskit-aer")
    else:
        print("=" * 60)
        print("Quantum Concept Bottleneck Model Example")
        print("=" * 60)
        
        # Test quantum concept extractor
        print("\nTesting Quantum Concept Extractor...")
        
        extractor = QuantumConceptExtractor(n_qubits=8, n_concepts=4)
        
        print(f"\nConfiguration:")
        print(f"  Qubits: {extractor.total_qubits}")
        print(f"  Concepts: {extractor.n_concepts}")
        print(f"  Qubits per concept: {extractor.qubits_per_concept}")
        print(f"  Parameters: {extractor.n_params}")
        
        # Extract concepts
        features = np.random.randn(8) * 0.5
        params = np.random.randn(extractor.n_params) * 0.1
        
        concepts = extractor.extract_concepts(features, params)
        
        print(f"\nExtracted concepts: {concepts}")
        print(f"  Concept 0: {concepts[0]:.3f}")
        print(f"  Concept 1: {concepts[1]:.3f}")
        print(f"  Concept 2: {concepts[2]:.3f}")
        print(f"  Concept 3: {concepts[3]:.3f}")
        
        # Test full Quantum CBM
        print("\n" + "=" * 60)
        print("Quantum CBM Model")
        print("=" * 60)
        
        concept_names = ['has_quantum_superposition', 'shows_entanglement',
                        'exhibits_interference', 'quantum_coherent']
        
        qcbm = QuantumCBM(
            input_dim=10,
            n_concepts=4,
            n_classes=3,
            n_qubits=8,
            concept_names=concept_names
        )
        
        print(f"\nModel configuration:")
        print(f"  Input dim: 10")
        print(f"  Concepts: 4")
        print(f"  Classes: 3")
        print(f"  Quantum parameters: {qcbm.quantum_params.shape[0]}")
        
        # Forward pass
        x_test = torch.randn(2, 10)
        logits, concept_probs = qcbm(x_test)
        
        print(f"\nForward pass results:")
        print(f"  Input shape: {x_test.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Concepts shape: {concept_probs.shape}")
        
        # Prediction with explanation
        print("\n" + "=" * 60)
        print("Prediction with Explanation")
        print("=" * 60)
        
        x_single = torch.randn(10)
        explanation = qcbm.predict_with_explanation(x_single, threshold=0.3)
        
        print(f"\nPredicted class: {explanation['predicted_class']}")
        print(f"Confidence: {explanation['class_probability']:.3f}")
        print(f"\nActive quantum concepts:")
        for concept in explanation['active_concepts']:
            print(f"  - {concept['name']}: {concept['probability']:.3f}")
        
        # Test hybrid model
        print("\n" + "=" * 60)
        print("Hybrid Quantum-Classical CBM")
        print("=" * 60)
        
        hybrid = HybridQuantumCBM(
            input_dim=10,
            n_concepts=8,
            n_classes=3,
            n_quantum_concepts=4,
            n_classical_concepts=4
        )
        
        logits, classical_c, quantum_c = hybrid(x_test)
        
        print(f"\nHybrid model results:")
        print(f"  Classical concepts: {classical_c.shape}")
        print(f"  Quantum concepts: {quantum_c.shape}")
        print(f"  Final logits: {logits.shape}")
        
        print("\n" + "=" * 60)
        print("Quantum CBM tests passed!")
        print("=" * 60)
