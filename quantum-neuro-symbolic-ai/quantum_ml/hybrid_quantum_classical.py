"""Hybrid Quantum-Classical Models.

Implements quantum circuits with classical optimization for machine learning.
Uses Qiskit for quantum simulation.
"""

import numpy as np
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from .quantum_backend import QuantumBackend, QuantumBackendConfig
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumBackend = None
    QuantumBackendConfig = None
    print("Warning: Qiskit not available. Install with: pip install qiskit qiskit-aer")

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Callable


class QuantumDataEncoder:
    """Encode classical data into quantum states."""
    
    @staticmethod
    def basis_encoding(data: np.ndarray, n_qubits: int) -> QuantumCircuit:
        """Basis encoding: map bit string to basis state.
        
        Args:
            data: Binary array of length n_qubits
            n_qubits: Number of qubits
            
        Returns:
            Quantum circuit encoding the data
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum encoding")
        
        qc = QuantumCircuit(n_qubits)
        
        for i, bit in enumerate(data[:n_qubits]):
            if bit:
                qc.x(i)
        
        return qc
    
    @staticmethod
    def amplitude_encoding(data: np.ndarray) -> QuantumCircuit:
        """Amplitude encoding: encode data in state amplitudes.
        
        Args:
            data: Data vector (will be normalized)
            
        Returns:
            Quantum circuit with data encoded in amplitudes
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum encoding")
        
        # Normalize data
        norm = np.linalg.norm(data)
        if norm > 0:
            normalized_data = data / norm
        else:
            normalized_data = data
        
        # Pad to power of 2
        n_qubits = int(np.ceil(np.log2(len(normalized_data))))
        padded_data = np.zeros(2**n_qubits)
        padded_data[:len(normalized_data)] = normalized_data
        
        # Create circuit
        qc = QuantumCircuit(n_qubits)
        qc.initialize(padded_data, range(n_qubits))
        
        return qc
    
    @staticmethod
    def angle_encoding(data: np.ndarray) -> QuantumCircuit:
        """Angle encoding: encode features as rotation angles.
        
        Args:
            data: Feature vector
            
        Returns:
            Quantum circuit with angle-encoded data
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum encoding")
        
        n_qubits = len(data)
        qc = QuantumCircuit(n_qubits)
        
        for i, value in enumerate(data):
            qc.ry(value, i)
        
        return qc


class ParameterizedQuantumCircuit:
    """Parameterized quantum circuit for variational algorithms."""
    
    def __init__(self, n_qubits: int, n_layers: int, entanglement: str = 'linear'):
        """Initialize parameterized circuit.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            entanglement: Entanglement pattern ('linear', 'full', 'circular')
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        
        # Calculate number of parameters
        self.n_params = n_qubits * n_layers * 3  # 3 rotations per qubit per layer
        
        # Create parameter vector
        self.params = ParameterVector('θ', self.n_params)
        
        # Build circuit
        self.circuit = self._build_circuit()
    
    def _build_circuit(self) -> QuantumCircuit:
        """Build the parameterized quantum circuit."""
        qc = QuantumCircuit(self.n_qubits)
        
        param_idx = 0
        
        for layer in range(self.n_layers):
            # Rotation layer (3 rotations per qubit)
            for qubit in range(self.n_qubits):
                qc.ry(self.params[param_idx], qubit)
                param_idx += 1
                qc.rz(self.params[param_idx], qubit)
                param_idx += 1
                qc.ry(self.params[param_idx], qubit)
                param_idx += 1
            
            # Entanglement layer
            if self.entanglement == 'linear':
                for qubit in range(self.n_qubits - 1):
                    qc.cx(qubit, qubit + 1)
            elif self.entanglement == 'circular':
                for qubit in range(self.n_qubits):
                    qc.cx(qubit, (qubit + 1) % self.n_qubits)
            elif self.entanglement == 'full':
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qc.cx(i, j)
        
        return qc
    
    def assign_parameters(self, param_values: np.ndarray) -> QuantumCircuit:
        """Bind parameter values to circuit.
        
        Args:
            param_values: Array of parameter values
            
        Returns:
            Bound quantum circuit
        """
        param_dict = {self.params[i]: param_values[i] for i in range(self.n_params)}
        return self.circuit.assign_parameters(param_dict)


class QuantumMeasurement:
    """Quantum measurement operations."""
    
    _cached_backend = None  # Cached QuantumBackend instance for local simulation
    
    @classmethod
    def get_backend(cls, use_hardware: bool = False) -> 'QuantumBackend':
        """Get cached backend instance.
        
        Args:
            use_hardware: If True, create hardware backend; if False, use cached local backend
            
        Returns:
            QuantumBackend instance
        """
        if use_hardware:
            # Always create new backend for hardware (don't cache)
            config = QuantumBackendConfig(use_hardware=True)
            return QuantumBackend(config)
        else:
            # Cache local simulator backend
            if cls._cached_backend is None:
                config = QuantumBackendConfig(use_hardware=False)
                cls._cached_backend = QuantumBackend(config)
            return cls._cached_backend
    
    @staticmethod
    def measure_expectation(circuit: QuantumCircuit, observable: str = 'Z', 
                          use_hardware: bool = False, shots: int = 1000) -> float:
        """Measure expectation value of an observable.
        
        Args:
            circuit: Quantum circuit to measure
            observable: Observable to measure ('Z', 'X', 'Y')
            use_hardware: If True, use IBM Quantum hardware
            shots: Number of measurement shots
            
        Returns:
            Expectation value
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        backend = QuantumMeasurement.get_backend(use_hardware)
        return backend.measure_expectation(circuit, observable, shots)
    
    @staticmethod
    def measure_probabilities(circuit: QuantumCircuit, shots: int = 1000,
                            use_hardware: bool = False) -> np.ndarray:
        """Measure probability distribution.
        
        Args:
            circuit: Quantum circuit
            shots: Number of measurement shots
            use_hardware: If True, use IBM Quantum hardware
            
        Returns:
            Probability array
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        backend = QuantumMeasurement.get_backend(use_hardware)
        return backend.measure_probabilities(circuit, shots)


class HybridQuantumClassicalModel:
    """Hybrid quantum-classical machine learning model."""
    
    def __init__(self, n_qubits: int, n_layers: int, n_classes: int,
                 backend: Optional['QuantumBackend'] = None):
        """Initialize hybrid model.
        
        Args:
            n_qubits: Number of qubits in quantum circuit
            n_layers: Number of variational layers
            n_classes: Number of output classes
            backend: Optional QuantumBackend for hardware execution
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.backend = backend
        
        # Quantum circuit
        self.quantum_circuit = ParameterizedQuantumCircuit(n_qubits, n_layers)
        
        # Parameters (randomly initialized)
        self.params = np.random.randn(self.quantum_circuit.n_params) * 0.1
        
        # Classical post-processing (simple linear layer)
        self.classical_weights = np.random.randn(n_qubits, n_classes) * 0.1
        self.classical_bias = np.zeros(n_classes)

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through hybrid model.
        
        Args:
            x: Input data (will be angle-encoded)
            
        Returns:
            Class logits
        """
        # Encode data
        encoding_circuit = QuantumDataEncoder.angle_encoding(x[:self.n_qubits])
        
        # Parameterized circuit
        param_circuit = self.quantum_circuit.assign_parameters(self.params)
        
        # Combine
        full_circuit = encoding_circuit.compose(param_circuit)
        
        # Measure each qubit in Z basis
        measurements = []
        if self.backend is not None:
            for qubit in range(self.n_qubits):
                meas_circuit = full_circuit.copy()
                expectation = self.backend.measure_expectation(meas_circuit, 'Z')
                measurements.append(expectation)
        else:
            for qubit in range(self.n_qubits):
                meas_circuit = full_circuit.copy()
                expectation = QuantumMeasurement.measure_expectation(meas_circuit, 'Z')
                measurements.append(expectation)
        
        measurements = np.array(measurements)
        
        # Classical post-processing
        logits = measurements @ self.classical_weights + self.classical_bias
        
        return logits
    
    def compute_loss(self, x: np.ndarray, y: int) -> float:
        """Compute cross-entropy loss.
        
        Args:
            x: Input features
            y: True label
            
        Returns:
            Loss value
        """
        logits = self.forward(x)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Cross-entropy
        loss = -np.log(probs[y] + 1e-10)
        
        return loss
    
    def parameter_shift_gradient(self, x: np.ndarray, y: int) -> np.ndarray:
        """Compute gradient using parameter shift rule.
        
        Args:
            x: Input features
            y: True label
            
        Returns:
            Gradient w.r.t. quantum parameters
        """
        gradients = np.zeros_like(self.params)
        shift = np.pi / 2
        original_params = self.params.copy()  # FIX: Save original once at start
        
        for i in range(len(self.params)):
            # Shift parameter +π/2
            self.params[i] = original_params[i] + shift
            loss_plus = self.compute_loss(x, y)
            
            # Shift parameter -π/2
            self.params[i] = original_params[i] - shift
            loss_minus = self.compute_loss(x, y)
            
            # Gradient
            gradients[i] = (loss_plus - loss_minus) / 2
            
            # Restore from original (FIX: eliminates array aliasing corruption)
            self.params[i] = original_params[i]
        
        return gradients


class QuantumNeuralNetwork(nn.Module):
    """PyTorch-compatible quantum neural network layer."""
    
    def __init__(self, n_qubits: int, n_layers: int, output_dim: int):
        super().__init__()
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_qubits = n_qubits
        self.quantum_circuit = ParameterizedQuantumCircuit(n_qubits, n_layers)
        
        # Quantum parameters (trainable)
        self.quantum_params = nn.Parameter(
            torch.randn(self.quantum_circuit.n_params) * 0.1
        )
        
        # Classical output layer
        self.classical_layer = nn.Linear(n_qubits, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_qubits)
            
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        batch_size = x.shape[0]
        quantum_outputs = []
        
        # Process each sample (quantum circuits are not batched)
        for i in range(batch_size):
            # WARNING: Gradient flow is SEVERED here - .detach().cpu().numpy() breaks PyTorch autograd
            # This means gradients will NOT flow back to quantum_params during backprop
            # To fix: migrate to PennyLane with interface="torch" for native autodiff
            x_i = x[i].detach().cpu().numpy()
            
            # Encode data
            encoding_circuit = QuantumDataEncoder.angle_encoding(x_i)
            
            # Parameterized circuit
            params_np = self.quantum_params.detach().cpu().numpy()
            param_circuit = self.quantum_circuit.assign_parameters(params_np)
            
            # Combine
            full_circuit = encoding_circuit.compose(param_circuit)
            
            # Measure expectations
            measurements = []
            for qubit in range(self.n_qubits):
                meas_circuit = full_circuit.copy()
                expectation = QuantumMeasurement.measure_expectation(meas_circuit, 'Z')
                measurements.append(expectation)
            
            quantum_outputs.append(measurements)
        
        quantum_outputs = torch.tensor(quantum_outputs, dtype=torch.float32)
        
        # Classical post-processing
        output = self.classical_layer(quantum_outputs)
        
        return output


if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Skipping examples.")
        print("Install with: pip install --break-system-packages qiskit qiskit-aer")
    else:
        print("=" * 60)
        print("Hybrid Quantum-Classical Models Example")
        print("=" * 60)
        
        # Test data encoding
        print("\nTesting quantum data encoding methods...")
        
        data = np.array([0.5, 0.3, 0.8])
        
        angle_circuit = QuantumDataEncoder.angle_encoding(data)
        print(f"\nAngle encoding circuit (3 qubits):")
        print(f"  Depth: {angle_circuit.depth()}")
        print(f"  Gates: {angle_circuit.count_ops()}")
        
        # Test parameterized circuit
        print("\n" + "=" * 60)
        print("Parameterized Quantum Circuit")
        print("=" * 60)
        
        pqc = ParameterizedQuantumCircuit(n_qubits=3, n_layers=2)
        print(f"\nCircuit properties:")
        print(f"  Qubits: {pqc.n_qubits}")
        print(f"  Layers: {pqc.n_layers}")
        print(f"  Parameters: {pqc.n_params}")
        print(f"  Depth: {pqc.circuit.depth()}")
        
        # Bind random parameters
        random_params = np.random.randn(pqc.n_params)
        bound_circuit = pqc.bind_parameters(random_params)
        print(f"\nBound circuit depth: {bound_circuit.depth()}")
        
        # Test hybrid model
        print("\n" + "=" * 60)
        print("Hybrid Quantum-Classical Model")
        print("=" * 60)
        
        model = HybridQuantumClassicalModel(n_qubits=4, n_layers=2, n_classes=3)
        
        print(f"\nModel configuration:")
        print(f"  Quantum qubits: {model.n_qubits}")
        print(f"  Variational layers: {model.n_layers}")
        print(f"  Output classes: {model.n_classes}")
        print(f"  Total quantum parameters: {len(model.params)}")
        
        # Test forward pass
        x_test = np.random.randn(4)
        print(f"\nTesting forward pass with input shape: {x_test.shape}")
        
        logits = model.forward(x_test)
        print(f"Output logits: {logits}")
        
        probs = np.exp(logits) / np.sum(np.exp(logits))
        print(f"Output probabilities: {probs}")
        print(f"Predicted class: {np.argmax(probs)}")
        
        print("\n" + "=" * 60)
        print("All hybrid quantum-classical tests passed!")
        print("=" * 60)
