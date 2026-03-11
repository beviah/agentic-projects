"""Variational Quantum Circuits and Algorithms.

Implements VQE, QAOA, and general variational quantum circuits.
"""

import numpy as np
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from .quantum_backend import QuantumBackend, QuantumBackendConfig
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumBackend = None
    QuantumBackendConfig = None

from typing import List, Tuple, Callable, Optional, Dict
from scipy.optimize import minimize


class Ansatz:
    """Quantum circuit ansatz for variational algorithms."""
    
    @staticmethod
    def hardware_efficient_ansatz(n_qubits: int, depth: int) -> Tuple[QuantumCircuit, ParameterVector]:
        """Hardware-efficient ansatz with alternating rotation and entanglement layers.
        
        Args:
            n_qubits: Number of qubits
            depth: Number of layers
            
        Returns:
            (circuit, parameters) tuple
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        n_params = n_qubits * depth * 3
        params = ParameterVector('θ', n_params)
        
        qc = QuantumCircuit(n_qubits)
        param_idx = 0
        
        for layer in range(depth):
            # Rotation layer
            for qubit in range(n_qubits):
                qc.rx(params[param_idx], qubit)
                param_idx += 1
                qc.ry(params[param_idx], qubit)
                param_idx += 1
                qc.rz(params[param_idx], qubit)
                param_idx += 1
            
            # Entanglement layer (linear connectivity)
            for qubit in range(n_qubits - 1):
                qc.cx(qubit, qubit + 1)
        
        return qc, params
    
    @staticmethod
    def real_amplitudes_ansatz(n_qubits: int, depth: int) -> Tuple[QuantumCircuit, ParameterVector]:
        """Real amplitudes ansatz using only RY rotations.
        
        Args:
            n_qubits: Number of qubits
            depth: Number of layers
            
        Returns:
            (circuit, parameters) tuple
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        n_params = n_qubits * depth
        params = ParameterVector('θ', n_params)
        
        qc = QuantumCircuit(n_qubits)
        param_idx = 0
        
        for layer in range(depth):
            # RY rotation layer
            for qubit in range(n_qubits):
                qc.ry(params[param_idx], qubit)
                param_idx += 1
            
            # Entanglement layer
            for qubit in range(n_qubits - 1):
                qc.cx(qubit, qubit + 1)
        
        return qc, params
    
    @staticmethod
    def qaoa_ansatz(n_qubits: int, p: int, cost_hamiltonian: SparsePauliOp) -> Tuple[QuantumCircuit, ParameterVector]:
        """QAOA ansatz for combinatorial optimization.
        
        Args:
            n_qubits: Number of qubits
            p: Number of QAOA layers
            cost_hamiltonian: Cost function Hamiltonian
            
        Returns:
            (circuit, parameters) tuple with 2*p parameters (gamma and beta)
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        params = ParameterVector('θ', 2 * p)
        
        qc = QuantumCircuit(n_qubits)
        
        # Initial state: equal superposition
        for qubit in range(n_qubits):
            qc.h(qubit)
        
        for layer in range(p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            
            # Cost Hamiltonian evolution (simplified for ZZ interactions)
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(2 * gamma, i + 1)
                qc.cx(i, i + 1)
            
            # Mixer Hamiltonian evolution (X rotations)
            for qubit in range(n_qubits):
                qc.rx(2 * beta, qubit)
        
        return qc, params


class VQE:
    """Variational Quantum Eigensolver.
    
    Finds ground state energy of a Hamiltonian using variational optimization.
    """
    
    def __init__(self, hamiltonian: SparsePauliOp, ansatz_type: str = 'hardware_efficient',
                 n_qubits: int = 4, depth: int = 2, backend: Optional[QuantumBackend] = None):
        """Initialize VQE.
        
        Args:
            hamiltonian: Target Hamiltonian
            ansatz_type: Type of ansatz ('hardware_efficient', 'real_amplitudes')
            n_qubits: Number of qubits
            depth: Ansatz depth
            backend: Optional QuantumBackend for hardware execution
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.backend = backend
        
        # Create ansatz
        if ansatz_type == 'hardware_efficient':
            self.ansatz, self.params = Ansatz.hardware_efficient_ansatz(n_qubits, depth)
        elif ansatz_type == 'real_amplitudes':
            self.ansatz, self.params = Ansatz.real_amplitudes_ansatz(n_qubits, depth)
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
        
        self.optimal_params = None
        self.optimal_energy = None
        self.history = []
    
    def expectation_value(self, param_values: np.ndarray) -> float:
        """Compute expectation value of Hamiltonian.
        
        Args:
            param_values: Parameter values
            
        Returns:
            <ψ(θ)|H|ψ(θ)>
        """
        # Bind parameters
        param_dict = {self.params[i]: param_values[i] for i in range(len(param_values))}
        bound_circuit = self.ansatz.assign_parameters(param_dict)
        
        if self.backend is None:
            # Local statevector simulation
            statevector = Statevector.from_instruction(bound_circuit)
            expectation = statevector.expectation_value(self.hamiltonian).real
        else:
            # Use backend for hardware-compatible execution
            expectations = self.backend.run_estimator([bound_circuit], [self.hamiltonian])
            expectation = expectations[0]
        
        return expectation
    
    def cost_function(self, param_values: np.ndarray) -> float:
        """Cost function for optimization (same as expectation)."""
        energy = self.expectation_value(param_values)
        self.history.append(energy)
        return energy
    
    def run(self, initial_params: Optional[np.ndarray] = None, 
            method: str = 'COBYLA', maxiter: int = 100) -> Dict:
        """Run VQE optimization.
        
        Args:
            initial_params: Initial parameter values (random if None)
            method: Optimization method
            maxiter: Maximum iterations
            
        Returns:
            Result dictionary
        """
        if initial_params is None:
            initial_params = np.random.randn(len(self.params)) * 0.1
        
        self.history = []
        
        # Wrap optimization in Session for hardware execution
        if self.backend is not None and self.backend.is_hardware:
            with self.backend.create_session():
                result = minimize(
                    self.cost_function,
                    initial_params,
                    method=method,
                    options={'maxiter': maxiter}
                )
        else:
            # Local simulation - no session needed
            result = minimize(
                self.cost_function,
                initial_params,
                method=method,
                options={'maxiter': maxiter}
            )
        
        self.optimal_params = result.x
        self.optimal_energy = result.fun
        
        return {
            'optimal_energy': self.optimal_energy,
            'optimal_params': self.optimal_params,
            'n_iterations': len(self.history),
            'history': self.history,
            'success': result.success
        }


class QAOA:
    """Quantum Approximate Optimization Algorithm.
    
    Solves combinatorial optimization problems.
    """
    
    def __init__(self, n_qubits: int, cost_function: Callable[[str], float], p: int = 1,
                 backend: Optional[QuantumBackend] = None):
        """Initialize QAOA.
        
        Args:
            n_qubits: Number of qubits
            cost_function: Classical cost function that takes bit string
            p: Number of QAOA layers
            backend: Optional QuantumBackend for hardware execution
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_qubits = n_qubits
        self.cost_function = cost_function
        self.p = p
        self.backend = backend
        
        # Create cost Hamiltonian (simplified)
        self.cost_hamiltonian = self._create_cost_hamiltonian()
        
        # Create ansatz
        self.ansatz, self.params = Ansatz.qaoa_ansatz(n_qubits, p, self.cost_hamiltonian)
        
        self.optimal_params = None
        self.optimal_solution = None
        self.history = []
    
    def _create_cost_hamiltonian(self) -> SparsePauliOp:
        """Create cost Hamiltonian (simplified ZZ interactions)."""
        # Simplified: use ZZ interactions between adjacent qubits
        pauli_list = []
        for i in range(self.n_qubits - 1):
            pauli_str = 'I' * i + 'ZZ' + 'I' * (self.n_qubits - i - 2)
            pauli_list.append((pauli_str, 1.0))
        
        return SparsePauliOp.from_list(pauli_list)
    
    def expectation_value(self, param_values: np.ndarray) -> float:
        """Compute expectation value of cost function.
        
        Args:
            param_values: QAOA parameters [gamma_1, beta_1, ..., gamma_p, beta_p]
            
        Returns:
            Expected cost
        """
        # Bind parameters
        param_dict = {self.params[i]: param_values[i] for i in range(len(param_values))}
        bound_circuit = self.ansatz.bind_parameters(param_dict)
        
        # Use provided backend or create temporary one
        shots = 1000
        if self.backend is not None:
            backend = self.backend
        else:
            backend = QuantumBackend(QuantumBackendConfig(use_hardware=False, shots=shots))
        
        results = backend.run_sampler([bound_circuit], shots=shots)
        counts = results[0]['counts']
        
        # Compute expected cost
        expected_cost = 0.0
        total_shots = sum(counts.values())
        for bitstring, count in counts.items():
            prob = count / total_shots
            cost = self.cost_function(bitstring)
            expected_cost += prob * cost
        
        return expected_cost
    
    def objective_function(self, param_values: np.ndarray) -> float:
        """Objective function for minimization."""
        cost = self.expectation_value(param_values)
        self.history.append(cost)
        return cost
    
    def run(self, initial_params: Optional[np.ndarray] = None, 
            method: str = 'COBYLA', maxiter: int = 100) -> Dict:
        """Run QAOA optimization.
        
        Args:
            initial_params: Initial parameters
            method: Optimization method
            maxiter: Maximum iterations
            
        Returns:
            Result dictionary
        """
        if initial_params is None:
            initial_params = np.random.randn(2 * self.p) * 0.1
        
        self.history = []
        
        # Wrap optimization in Session for hardware execution
        if self.backend is not None and self.backend.is_hardware:
            with self.backend.create_session():
                result = minimize(
                    self.objective_function,
                    initial_params,
                    method=method,
                    options={'maxiter': maxiter}
                )
        else:
            # Local simulation - no session needed
            result = minimize(
                self.objective_function,
                initial_params,
                method=method,
                options={'maxiter': maxiter}
            )
        
        self.optimal_params = result.x
        
        # Get best solution
        param_dict = {self.params[i]: self.optimal_params[i] for i in range(len(self.optimal_params))}
        bound_circuit = self.ansatz.bind_parameters(param_dict)
        
        # Use provided backend or create temporary one for measurement
        if self.backend is not None:
            backend = self.backend
        else:
            backend_config = QuantumBackendConfig(use_hardware=False, shots=1000)
            backend = QuantumBackend(backend_config)
        
        results = backend.run_sampler([bound_circuit], shots=1000)
        counts = results[0]['counts']
        
        # Most frequent bitstring
        self.optimal_solution = max(counts, key=counts.get)
        
        return {
            'optimal_solution': self.optimal_solution,
            'optimal_cost': self.cost_function(self.optimal_solution),
            'optimal_params': self.optimal_params,
            'n_iterations': len(self.history),
            'history': self.history,
            'success': result.success
        }


class VariationalQuantumClassifier:
    """Variational quantum circuit for classification."""
    
    def __init__(self, n_qubits: int, n_layers: int, n_classes: int,
                 backend: Optional[QuantumBackend] = None):
        """Initialize VQC classifier.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            n_classes: Number of classes
            backend: Optional QuantumBackend for hardware execution
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        
        # Backend (cached - created once, not per forward call)
        if backend is not None:
            self._backend = backend
        else:
            self._backend = QuantumBackend(QuantumBackendConfig(use_hardware=False, shots=100))
        
        # Create ansatz
        self.ansatz, self.params = Ansatz.hardware_efficient_ansatz(n_qubits, n_layers)
        
        # Parameters (randomly initialized)
        self.param_values = np.random.randn(len(self.params)) * 0.1
        
        # Classical readout weights
        self.readout_weights = np.random.randn(n_qubits, n_classes) * 0.1

    
    def encode_data(self, x: np.ndarray) -> QuantumCircuit:
        """Encode classical data into quantum state.
        
        Args:
            x: Input features
            
        Returns:
            Encoding circuit
        """
        qc = QuantumCircuit(self.n_qubits)
        
        for i in range(min(self.n_qubits, len(x))):
            qc.ry(x[i], i)
        
        return qc
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Class logits
        """
        # Encode data
        encoding = self.encode_data(x)
        
        # Apply parameterized circuit
        param_dict = {self.params[i]: self.param_values[i] for i in range(len(self.params))}
        variational = self.ansatz.assign_parameters(param_dict)
        
        # Combine
        full_circuit = encoding.compose(variational)
        
        # Measure expectation values using cached backend
        measurements = []
        for qubit in range(self.n_qubits):
            # Measure Z expectation for each qubit
            qc_copy = full_circuit.copy()
            results = self._backend.run_sampler([qc_copy], shots=100)
            counts = results[0]['counts']
            
            # Compute expectation of Z_i
            expectation = 0.0
            for bitstring, count in counts.items():
                bit_value = int(bitstring[-(qubit+1)])
                parity = 1 if bit_value == 0 else -1
                expectation += parity * count / 100
            
            measurements.append(expectation)
        
        measurements = np.array(measurements)
        
        # Classical readout
        logits = measurements @ self.readout_weights
        
        return logits
    
    def predict(self, x: np.ndarray) -> int:
        """Predict class.
        
        Args:
            x: Input features
            
        Returns:
            Predicted class
        """
        logits = self.forward(x)
        return np.argmax(logits)


if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Skipping examples.")
        print("Install with: pip install --break-system-packages qiskit qiskit-aer")
    else:
        print("=" * 60)
        print("Variational Quantum Circuits Example")
        print("=" * 60)
        
        # Test ansatz creation
        print("\nTesting ansatz creation...")
        
        hw_ansatz, hw_params = Ansatz.hardware_efficient_ansatz(n_qubits=4, depth=2)
        print(f"\nHardware-efficient ansatz:")
        print(f"  Qubits: {hw_ansatz.num_qubits}")
        print(f"  Depth: {hw_ansatz.depth()}")
        print(f"  Parameters: {len(hw_params)}")
        print(f"  Gates: {hw_ansatz.count_ops()}")
        
        ra_ansatz, ra_params = Ansatz.real_amplitudes_ansatz(n_qubits=4, depth=2)
        print(f"\nReal amplitudes ansatz:")
        print(f"  Qubits: {ra_ansatz.num_qubits}")
        print(f"  Depth: {ra_ansatz.depth()}")
        print(f"  Parameters: {len(ra_params)}")
        
        # Test VQE
        print("\n" + "=" * 60)
        print("Variational Quantum Eigensolver (VQE)")
        print("=" * 60)
        
        # Create simple Hamiltonian: H = Z_0 + Z_1 (ground state energy = -2)
        hamiltonian = SparsePauliOp.from_list([('IIZZ', 1.0), ('ZZII', 1.0)])
        
        print(f"\nTarget Hamiltonian: Z_0 + Z_1")
        print(f"Expected ground state energy: -2.0")
        
        vqe = VQE(hamiltonian, ansatz_type='hardware_efficient', n_qubits=4, depth=1)
        
        print(f"\nRunning VQE optimization...")
        result = vqe.run(maxiter=50)
        
        print(f"\nVQE Results:")
        print(f"  Optimal energy: {result['optimal_energy']:.4f}")
        print(f"  Iterations: {result['n_iterations']}")
        print(f"  Success: {result['success']}")
        print(f"  Energy improvement: {result['history'][0]:.4f} -> {result['history'][-1]:.4f}")
        
        # Test QAOA
        print("\n" + "=" * 60)
        print("Quantum Approximate Optimization Algorithm (QAOA)")
        print("=" * 60)
        
        # Simple cost function: prefer bitstrings with even number of 1s
        def simple_cost(bitstring: str) -> float:
            num_ones = bitstring.count('1')
            return abs(num_ones - 2)  # Prefer exactly 2 ones
        
        print(f"\nCost function: Prefer bitstrings with exactly 2 ones")
        print(f"For 4 qubits, optimal solutions: 0011, 0101, 0110, 1001, 1010, 1100")
        
        qaoa = QAOA(n_qubits=4, cost_function=simple_cost, p=1)
        
        print(f"\nRunning QAOA optimization (p=1)...")
        result = qaoa.run(maxiter=30)
        
        print(f"\nQAOA Results:")
        print(f"  Optimal solution: {result['optimal_solution']}")
        print(f"  Optimal cost: {result['optimal_cost']}")
        print(f"  Iterations: {result['n_iterations']}")
        print(f"  Success: {result['success']}")
        
        # Test VQC Classifier
        print("\n" + "=" * 60)
        print("Variational Quantum Classifier")
        print("=" * 60)
        
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=1, n_classes=2)
        
        print(f"\nClassifier configuration:")
        print(f"  Qubits: {vqc.n_qubits}")
        print(f"  Classes: {vqc.n_classes}")
        print(f"  Parameters: {len(vqc.param_values)}")
        
        # Test prediction
        x_test = np.array([0.5, 0.3, 0.8, 0.2])
        print(f"\nTest input: {x_test}")
        
        logits = vqc.forward(x_test)
        prediction = vqc.predict(x_test)
        
        print(f"Logits: {logits}")
        print(f"Predicted class: {prediction}")
        
        print("\n" + "=" * 60)
        print("Variational quantum circuits tests passed!")
        print("=" * 60)
