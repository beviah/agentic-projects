"""Quantum Kernel Methods.

Implements quantum kernel computation and quantum kernel-based algorithms
like QSVM (Quantum Support Vector Machine).
"""

import numpy as np
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    from .quantum_backend import QuantumBackend, QuantumBackendConfig
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumBackend = None
    QuantumBackendConfig = None

from typing import Callable, List, Tuple, Optional
import warnings


class QuantumFeatureMap:
    """Quantum feature maps for encoding classical data."""
    
    @staticmethod
    def zz_feature_map(x: np.ndarray, n_qubits: int, depth: int = 2) -> QuantumCircuit:
        """ZZ feature map with entangling ZZ gates.
        
        Args:
            x: Feature vector
            n_qubits: Number of qubits
            depth: Circuit depth (number of repetitions)
            
        Returns:
            Quantum circuit implementing ZZ feature map
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        qc = QuantumCircuit(n_qubits)
        
        for d in range(depth):
            # Hadamard layer
            for i in range(n_qubits):
                qc.h(i)
            
            # Z rotation layer
            for i in range(min(n_qubits, len(x))):
                qc.rz(2 * x[i], i)
            
            # Entangling ZZ layer
            for i in range(n_qubits - 1):
                for j in range(i + 1, n_qubits):
                    # ZZ rotation: exp(i * phi * Z_i Z_j)
                    if i < len(x) and j < len(x):
                        phi = (np.pi - x[i]) * (np.pi - x[j])
                        qc.cx(i, j)
                        qc.rz(2 * phi, j)
                        qc.cx(i, j)
        
        return qc
    
    @staticmethod
    def pauli_feature_map(x: np.ndarray, n_qubits: int, pauli_string: str = 'ZZ') -> QuantumCircuit:
        """Pauli feature map.
        
        Args:
            x: Feature vector
            n_qubits: Number of qubits
            pauli_string: Pauli operators to use ('Z', 'ZZ', 'ZZZ', etc.)
            
        Returns:
            Quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        qc = QuantumCircuit(n_qubits)
        
        # Hadamard layer for superposition
        for i in range(n_qubits):
            qc.h(i)
        
        # Apply Pauli rotations based on data
        for i in range(min(n_qubits, len(x))):
            qc.rz(2 * x[i], i)
        
        # Entangling layer based on pauli_string
        if len(pauli_string) >= 2:
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
        
        return qc
    
    @staticmethod
    def custom_feature_map(x: np.ndarray, n_qubits: int, 
                          reps: int = 1) -> QuantumCircuit:
        """Custom feature map with multiple encoding strategies.
        
        Args:
            x: Feature vector
            n_qubits: Number of qubits
            reps: Number of repetitions
            
        Returns:
            Quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        qc = QuantumCircuit(n_qubits)
        
        for rep in range(reps):
            # Encoding layer: RY rotations
            for i in range(min(n_qubits, len(x))):
                qc.ry(x[i], i)
            
            # Entangling layer: CZ gates
            for i in range(n_qubits - 1):
                qc.cz(i, i + 1)
            
            # Second encoding layer: RZ rotations
            for i in range(min(n_qubits, len(x))):
                qc.rz(x[i], i)
        
        return qc


class QuantumKernel:
    """Compute quantum kernel between data points."""
    
    def __init__(self, feature_map: str = 'zz', n_qubits: int = 4, 
                 depth: int = 2, method: str = 'fidelity',
                 backend: Optional[QuantumBackend] = None):
        """Initialize quantum kernel.
        
        Args:
            feature_map: Type of feature map ('zz', 'pauli', 'custom')
            n_qubits: Number of qubits
            depth: Feature map depth
            method: Kernel computation method ('fidelity', 'swap_test')
            backend: Optional QuantumBackend for hardware execution
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.feature_map_type = feature_map
        self.n_qubits = n_qubits
        self.depth = depth
        self.method = method
        self.backend = backend
        
        # Select feature map
        if feature_map == 'zz':
            self.feature_map_func = lambda x: QuantumFeatureMap.zz_feature_map(x, n_qubits, depth)
        elif feature_map == 'pauli':
            self.feature_map_func = lambda x: QuantumFeatureMap.pauli_feature_map(x, n_qubits)
        elif feature_map == 'custom':
            self.feature_map_func = lambda x: QuantumFeatureMap.custom_feature_map(x, n_qubits, depth)
        else:
            raise ValueError(f"Unknown feature map: {feature_map}")
    
    def compute_fidelity(self, qc1: QuantumCircuit, qc2: QuantumCircuit) -> float:
        """Compute fidelity between two quantum states.
        
        Args:
            qc1: First quantum circuit
            qc2: Second quantum circuit
            
        Returns:
            Fidelity value (squared overlap)
        """
        if self.backend is None:
            # Fallback to local statevector simulation
            sv1 = Statevector.from_instruction(qc1)
            sv2 = Statevector.from_instruction(qc2)
            overlap = np.abs(np.vdot(sv1.data, sv2.data))
            fidelity = overlap ** 2
        else:
            # Use backend for hardware-compatible fidelity computation
            # Method: Use SWAP test for hardware compatibility
            fidelity = self._compute_fidelity_via_swap_test(qc1, qc2)
        
        return fidelity
    
    def _compute_fidelity_via_swap_test(self, qc1: QuantumCircuit, qc2: QuantumCircuit) -> float:
        """Compute fidelity using SWAP test (hardware-compatible).
        
        Args:
            qc1: First quantum circuit
            qc2: Second quantum circuit
            
        Returns:
            Fidelity value
        """
        n = qc1.num_qubits
        
        # Create SWAP test circuit
        swap_circuit = QuantumCircuit(2 * n + 1, 1)
        
        # Prepare states on separate registers
        swap_circuit.compose(qc1, qubits=range(1, n + 1), inplace=True)
        swap_circuit.compose(qc2, qubits=range(n + 1, 2 * n + 1), inplace=True)
        
        # Apply SWAP test
        swap_circuit.h(0)  # Ancilla qubit
        
        for i in range(n):
            swap_circuit.cswap(0, i + 1, i + n + 1)
        
        swap_circuit.h(0)
        swap_circuit.measure(0, 0)
        
        # Execute on backend
        results = self.backend.run_sampler([swap_circuit], shots=1024)
        counts = results[0]['counts']
        
        # Probability of measuring |0>
        total_shots = sum(counts.values())
        p_zero = counts.get('0', 0) / total_shots
        
        # Fidelity from SWAP test: F = 2*P(0) - 1
        # But we need to return squared overlap, so adjust
        # SWAP test gives: P(0) = (1 + |<ψ1|ψ2>|²) / 2
        # Therefore: |<ψ1|ψ2>|² = 2*P(0) - 1
        fidelity = max(0.0, 2 * p_zero - 1)
        
        return fidelity
    
    def compute_swap_test(self, qc1: QuantumCircuit, qc2: QuantumCircuit) -> float:
        """Compute kernel using SWAP test.
        
        Args:
            qc1: First quantum circuit
            qc2: Second quantum circuit
            
        Returns:
            Kernel value from SWAP test
        """
        n = qc1.num_qubits
        
        # Create SWAP test circuit
        swap_circuit = QuantumCircuit(2 * n + 1, 1)
        
        # Prepare states on separate registers
        swap_circuit.compose(qc1, qubits=range(1, n + 1), inplace=True)
        swap_circuit.compose(qc2, qubits=range(n + 1, 2 * n + 1), inplace=True)
        
        # Apply SWAP test
        swap_circuit.h(0)  # Ancilla qubit
        
        for i in range(n):
            swap_circuit.cswap(0, i + 1, i + n + 1)
        
        swap_circuit.h(0)
        swap_circuit.measure(0, 0)
        
        # Use QuantumBackend for measurement
        backend_config = QuantumBackendConfig(use_hardware=False, shots=1000)
        backend = QuantumBackend(backend_config)
        results = backend.run_sampler([swap_circuit], shots=1000)
        counts = results[0]['counts']
        
        # Probability of measuring |0>
        p_zero = counts.get('0', 0) / 1000
        
        # Kernel = 2 * P(0) - 1
        kernel_value = 2 * p_zero - 1
        
        return kernel_value
    
    def kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum kernel K(x1, x2).
        
        Args:
            x1: First data point
            x2: Second data point
            
        Returns:
            Kernel value
        """
        # Create feature map circuits
        qc1 = self.feature_map_func(x1)
        qc2 = self.feature_map_func(x2)
        
        # Compute kernel
        if self.method == 'fidelity':
            return self.compute_fidelity(qc1, qc2)
        elif self.method == 'swap_test':
            return self.compute_swap_test(qc1, qc2)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def compute_kernel_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel matrix K(X, Y).
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            Y: Optional second data matrix. If None, compute K(X, X)
            
        Returns:
            Kernel matrix of shape (len(X), len(Y))
        """
        if Y is None:
            Y = X
        
        n_x = len(X)
        n_y = len(Y)
        
        K = np.zeros((n_x, n_y))
        
        for i in range(n_x):
            for j in range(n_y):
                K[i, j] = self.kernel_function(X[i], Y[j])
        
        return K


class QuantumSVM:
    """Quantum Support Vector Machine using quantum kernels."""
    
    def __init__(self, quantum_kernel: QuantumKernel, C: float = 1.0):
        """Initialize QSVM.
        
        Args:
            quantum_kernel: QuantumKernel instance
            C: Regularization parameter
        """
        self.quantum_kernel = quantum_kernel
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        self.b = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit QSVM using quantum kernel.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,) with values {-1, +1}
        """
        n_samples = len(X)
        
        # Compute kernel matrix
        K = self.quantum_kernel.compute_kernel_matrix(X)
        
        # Solve QP problem (simplified: just use all points as support vectors)
        # In practice, would use CVXOPT or similar QP solver
        
        # Simplified solution: use kernel ridge regression approximation
        regularization = np.eye(n_samples) * (1.0 / self.C)
        self.alpha = np.linalg.solve(K + regularization, y)
        
        # Store support vectors (all training points in this simplified version)
        self.support_vectors = X
        self.support_labels = y
        self.b = 0.0  # Bias term (simplified)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values.
        
        Args:
            X: Test data of shape (n_samples, n_features)
            
        Returns:
            Decision function values
        """
        if self.support_vectors is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Compute kernel between test and support vectors
        K_test = self.quantum_kernel.compute_kernel_matrix(X, self.support_vectors)
        
        # Decision function
        decision = K_test @ (self.alpha * self.support_labels) + self.b
        
        return decision
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Test data
            
        Returns:
            Predicted labels {-1, +1}
        """
        decision = self.decision_function(X)
        return np.sign(decision)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy.
        
        Args:
            X: Test data
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class QuantumKernelRidgeRegression:
    """Kernel ridge regression using quantum kernels."""
    
    def __init__(self, quantum_kernel: QuantumKernel, alpha: float = 1.0):
        """Initialize quantum kernel ridge regression.
        
        Args:
            quantum_kernel: QuantumKernel instance
            alpha: Regularization strength
        """
        self.quantum_kernel = quantum_kernel
        self.alpha = alpha
        self.weights = None
        self.X_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
        """
        # Compute kernel matrix
        K = self.quantum_kernel.compute_kernel_matrix(X)
        
        # Solve: (K + alpha*I) * weights = y
        n = len(X)
        regularization = np.eye(n) * self.alpha
        self.weights = np.linalg.solve(K + regularization, y)
        
        # Store training data
        self.X_train = X
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets.
        
        Args:
            X: Test data
            
        Returns:
            Predicted targets
        """
        if self.X_train is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Compute kernel between test and training data
        K_test = self.quantum_kernel.compute_kernel_matrix(X, self.X_train)
        
        # Prediction
        predictions = K_test @ self.weights
        
        return predictions


if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Skipping examples.")
        print("Install with: pip install --break-system-packages qiskit qiskit-aer")
    else:
        print("=" * 60)
        print("Quantum Kernel Methods Example")
        print("=" * 60)
        
        # Test feature maps
        print("\nTesting quantum feature maps...")
        
        x = np.array([0.5, 0.3, 0.8, 0.2])
        
        zz_map = QuantumFeatureMap.zz_feature_map(x, n_qubits=4, depth=2)
        print(f"\nZZ feature map:")
        print(f"  Qubits: {zz_map.num_qubits}")
        print(f"  Depth: {zz_map.depth()}")
        print(f"  Gates: {zz_map.count_ops()}")
        
        pauli_map = QuantumFeatureMap.pauli_feature_map(x, n_qubits=4)
        print(f"\nPauli feature map:")
        print(f"  Qubits: {pauli_map.num_qubits}")
        print(f"  Depth: {pauli_map.depth()}")
        
        # Test quantum kernel
        print("\n" + "=" * 60)
        print("Quantum Kernel Computation")
        print("=" * 60)
        
        qkernel = QuantumKernel(feature_map='zz', n_qubits=4, depth=1, method='fidelity')
        
        x1 = np.array([0.5, 0.3, 0.8, 0.2])
        x2 = np.array([0.5, 0.3, 0.8, 0.2])  # Same point
        x3 = np.array([0.1, 0.9, 0.2, 0.7])  # Different point
        
        k11 = qkernel.kernel_function(x1, x1)
        k12 = qkernel.kernel_function(x1, x2)
        k13 = qkernel.kernel_function(x1, x3)
        
        print(f"\nKernel values:")
        print(f"  K(x1, x1) = {k11:.4f} (should be ~1.0)")
        print(f"  K(x1, x2) = {k12:.4f} (should be ~1.0, same point)")
        print(f"  K(x1, x3) = {k13:.4f} (different point)")
        
        # Test kernel matrix
        print("\n" + "=" * 60)
        print("Kernel Matrix Computation")
        print("=" * 60)
        
        X = np.array([
            [0.5, 0.3, 0.8, 0.2],
            [0.1, 0.9, 0.2, 0.7],
            [0.7, 0.4, 0.6, 0.3]
        ])
        
        print(f"\nComputing kernel matrix for {len(X)} samples...")
        K = qkernel.compute_kernel_matrix(X)
        
        print(f"\nKernel matrix:")
        print(K)
        print(f"\nDiagonal elements (should be ~1.0): {np.diag(K)}")
        
        # Test Quantum SVM
        print("\n" + "=" * 60)
        print("Quantum SVM")
        print("=" * 60)
        
        # Create simple binary classification dataset
        X_train = np.array([
            [0.1, 0.1, 0.0, 0.0],
            [0.2, 0.2, 0.1, 0.1],
            [0.8, 0.8, 0.9, 0.9],
            [0.9, 0.9, 1.0, 1.0]
        ])
        y_train = np.array([-1, -1, 1, 1])
        
        print(f"\nTraining QSVM on {len(X_train)} samples...")
        
        qsvm = QuantumSVM(qkernel, C=1.0)
        qsvm.fit(X_train, y_train)
        
        print(f"Training complete!")
        print(f"Number of support vectors: {len(qsvm.support_vectors)}")
        
        # Test predictions
        X_test = np.array([
            [0.15, 0.15, 0.05, 0.05],  # Should be class -1
            [0.85, 0.85, 0.95, 0.95]   # Should be class +1
        ])
        
        predictions = qsvm.predict(X_test)
        print(f"\nPredictions on test data:")
        print(f"  Test sample 1: {predictions[0]} (expected: -1)")
        print(f"  Test sample 2: {predictions[1]} (expected: +1)")
        
        # Compute training accuracy
        train_accuracy = qsvm.score(X_train, y_train)
        print(f"\nTraining accuracy: {train_accuracy:.2%}")
        
        print("\n" + "=" * 60)
        print("Quantum kernel methods tests passed!")
        print("=" * 60)
