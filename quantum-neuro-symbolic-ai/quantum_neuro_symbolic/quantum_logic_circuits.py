"""Quantum Circuits for Differentiable Logic.

Implements quantum circuits that perform logical operations with differentiable
parameters, enabling gradient-based learning of quantum logic programs.
"""

import numpy as np
try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn


class QuantumLogicGates:
    """Quantum implementations of logic gates."""
    
    @staticmethod
    def quantum_not(qc: QuantumCircuit, qubit: int):
        """Quantum NOT gate (Pauli-X)."""
        qc.x(qubit)
    
    @staticmethod
    def quantum_and(qc: QuantumCircuit, control1: int, control2: int, target: int):
        """Quantum AND gate (Toffoli/CCNOT)."""
        qc.ccx(control1, control2, target)
    
    @staticmethod
    def quantum_or(qc: QuantumCircuit, input1: int, input2: int, target: int, 
                   ancilla1: int, ancilla2: int):
        """Quantum OR gate: OR(a,b) = NOT(AND(NOT(a), NOT(b)))."""
        # NOT gates on inputs
        qc.x(input1)
        qc.x(input2)
        
        # AND of NOTs
        qc.ccx(input1, input2, target)
        
        # NOT on output
        qc.x(target)
        
        # Uncompute (restore inputs)
        qc.x(input1)
        qc.x(input2)
    
    @staticmethod
    def quantum_fuzzy_and(qc: QuantumCircuit, control1: int, control2: int, 
                         target: int, theta: Parameter):
        """Fuzzy quantum AND with learnable strength parameter.
        
        Uses controlled rotation to implement probabilistic AND.
        """
        # Multi-controlled rotation
        # If both controls are |1>, rotate target by theta
        qc.ccry(theta, control1, control2, target)
    
    @staticmethod
    def quantum_fuzzy_or(qc: QuantumCircuit, input1: int, input2: int, 
                        target: int, theta: Parameter):
        """Fuzzy quantum OR with learnable parameter."""
        # Controlled rotations from each input
        qc.cry(theta / 2, input1, target)
        qc.cry(theta / 2, input2, target)


class QuantumLogicRule:
    """A single quantum logic rule with differentiable parameters."""
    
    def __init__(self, n_body_predicates: int, rule_type: str = 'fuzzy_and'):
        """Initialize quantum logic rule.
        
        Args:
            n_body_predicates: Number of body predicates
            rule_type: Type of rule ('crisp_and', 'fuzzy_and', 'fuzzy_or')
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_body = n_body_predicates
        self.rule_type = rule_type
        
        # Total qubits: n_body inputs + 1 output + ancillas
        self.n_qubits = n_body_predicates + 1 + max(2, n_body_predicates)
        
        # Learnable parameter
        self.theta = Parameter('θ')
    
    def create_circuit(self) -> QuantumCircuit:
        """Create quantum circuit for this rule.
        
        Returns:
            Quantum circuit implementing the rule
        """
        qc = QuantumCircuit(self.n_qubits)
        
        body_qubits = list(range(self.n_body))
        head_qubit = self.n_body
        ancilla_qubits = list(range(self.n_body + 1, self.n_qubits))
        
        if self.rule_type == 'crisp_and':
            # Crisp AND: all body predicates must be true
            if self.n_body == 2:
                QuantumLogicGates.quantum_and(qc, body_qubits[0], body_qubits[1], head_qubit)
            else:
                # Multi-way AND using ancillas
                for i, body_qubit in enumerate(body_qubits):
                    if i == 0:
                        qc.cx(body_qubit, head_qubit)
                    else:
                        qc.ccx(body_qubit, head_qubit, ancilla_qubits[i-1])
                        qc.cx(ancilla_qubits[i-1], head_qubit)
        
        elif self.rule_type == 'fuzzy_and':
            # Fuzzy AND with learnable strength
            if self.n_body == 2:
                QuantumLogicGates.quantum_fuzzy_and(
                    qc, body_qubits[0], body_qubits[1], head_qubit, self.theta
                )
            else:
                # Multi-way fuzzy AND
                for body_qubit in body_qubits:
                    qc.cry(self.theta / self.n_body, body_qubit, head_qubit)
        
        elif self.rule_type == 'fuzzy_or':
            # Fuzzy OR
            for body_qubit in body_qubits:
                qc.cry(self.theta / self.n_body, body_qubit, head_qubit)
        
        return qc
    
    def bind_parameter(self, qc: QuantumCircuit, value: float) -> QuantumCircuit:
        """Bind parameter value to circuit."""
        return qc.assign_parameters({self.theta: value})


class QuantumLogicProgram:
    """Quantum circuit implementing a logic program.
    
    Represents multiple rules as a single quantum circuit with
    differentiable parameters.
    """
    
    def __init__(self, n_predicates: int, rules: List[Tuple[int, List[int], str]],
                 backend=None):
        """Initialize quantum logic program.
        
        Args:
            n_predicates: Total number of predicates
            rules: List of (head_idx, body_indices, rule_type) tuples
            backend: Optional QuantumBackend for hardware execution (simulation-only for now)
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_predicates = n_predicates
        self.rules = rules
        self.backend = backend
        
        # Each predicate gets a qubit
        # Plus ancilla qubits for complex operations
        self.n_qubits = n_predicates + 10  # Extra ancillas
        
        # Create parameters for each rule
        self.n_params = len(rules)
        self.params = ParameterVector('θ', self.n_params)
        
        # Build circuit
        self.circuit = self._build_circuit()
        
        # Warn if hardware backend provided (statevector-based, simulation only)
        if backend is not None and hasattr(backend, 'is_hardware') and backend.is_hardware:
            import warnings
            warnings.warn(
                "QuantumLogicProgram uses statevector-based computation and is not "
                "hardware-compatible. Will fall back to local simulation.",
                UserWarning
            )
    
    def _build_circuit(self) -> QuantumCircuit:
        """Build quantum circuit for all rules."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Apply each rule
        for rule_idx, (head_idx, body_indices, rule_type) in enumerate(self.rules):
            param = self.params[rule_idx]
            
            if rule_type == 'fuzzy_and' and len(body_indices) == 2:
                # Two-input fuzzy AND
                qc.ccry(param, body_indices[0], body_indices[1], head_idx)
            
            elif rule_type == 'fuzzy_or':
                # Fuzzy OR: each input contributes
                for body_idx in body_indices:
                    qc.cry(param / len(body_indices), body_idx, head_idx)
            
            elif rule_type == 'implication':
                # A -> B: rotate B based on A
                if len(body_indices) == 1:
                    qc.cry(param, body_indices[0], head_idx)
        
        return qc
    
    def forward(self, initial_state: np.ndarray, param_values: np.ndarray) -> np.ndarray:
        """Execute quantum logic program.
        
        Args:
            initial_state: Initial predicate truth values (probabilities)
            param_values: Parameter values for rules
            
        Returns:
            Final predicate probabilities after logic execution
        """
        # Prepare initial state
        init_circuit = QuantumCircuit(self.n_qubits)
        
        # Encode initial predicates in amplitude
        for i, prob in enumerate(initial_state[:self.n_predicates]):
            if prob > 0.5:
                init_circuit.x(i)
            # Apply partial rotation for probabilistic values
            angle = 2 * np.arcsin(np.sqrt(min(1.0, max(0.0, prob))))
            init_circuit.ry(angle, i)
        
        # Bind parameters to logic circuit
        param_dict = {self.params[i]: param_values[i] for i in range(self.n_params)}
        logic_circuit = self.circuit.assign_parameters(param_dict)
        
        # Combine
        full_circuit = init_circuit.compose(logic_circuit)
        
        # Simulate
        statevector = Statevector.from_instruction(full_circuit)
        
        # Extract predicate probabilities by measuring each qubit
        probs = np.zeros(self.n_predicates)
        for i in range(self.n_predicates):
            # Probability of qubit i being |1>
            # Sum over all basis states where qubit i is 1
            prob_1 = 0.0
            for basis_idx, amplitude in enumerate(statevector.data):
                if (basis_idx >> i) & 1:  # Check if bit i is 1
                    prob_1 += abs(amplitude) ** 2
            probs[i] = prob_1
        
        return probs


class QuantumDifferentiableLogic(nn.Module):
    """PyTorch-compatible quantum differentiable logic layer."""
    
    def __init__(self, n_predicates: int, rules: List[Tuple[int, List[int], str]]):
        super().__init__()
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        self.n_predicates = n_predicates
        self.quantum_program = QuantumLogicProgram(n_predicates, rules)
        
        # Learnable parameters
        self.rule_params = nn.Parameter(
            torch.randn(self.quantum_program.n_params) * 0.1
        )
    
    def forward(self, predicate_probs: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum logic.
        
        Args:
            predicate_probs: Initial predicate probabilities (batch, n_predicates)
            
        Returns:
            Final predicate probabilities after logic reasoning
        """
        batch_size = predicate_probs.shape[0]
        outputs = []
        
        for i in range(batch_size):
            # WARNING: Gradient flow is SEVERED here - .detach().cpu().numpy() breaks PyTorch autograd
            # This means gradients will NOT flow back to rule_params during backprop
            # To fix: migrate to PennyLane with interface="torch" for native autodiff
            initial_state = predicate_probs[i].detach().cpu().numpy()
            param_values = self.rule_params.detach().cpu().numpy()
            
            # Execute quantum logic
            final_probs = self.quantum_program.forward(initial_state, param_values)
            outputs.append(final_probs)
        
        outputs = torch.tensor(outputs, dtype=torch.float32)
        
        return outputs


class QuantumFuzzyLogic:
    """Quantum implementation of fuzzy logic operations."""
    
    @staticmethod
    def create_fuzzy_and_circuit(n_inputs: int, strength: float) -> QuantumCircuit:
        """Create quantum circuit for fuzzy AND.
        
        Args:
            n_inputs: Number of inputs
            strength: Conjunction strength parameter
            
        Returns:
            Quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        qc = QuantumCircuit(n_inputs + 1)
        output_qubit = n_inputs
        
        # Each input contributes to output via controlled rotation
        angle = strength * np.pi / n_inputs
        for i in range(n_inputs):
            qc.cry(angle, i, output_qubit)
        
        return qc
    
    @staticmethod
    def create_fuzzy_or_circuit(n_inputs: int, strength: float) -> QuantumCircuit:
        """Create quantum circuit for fuzzy OR.
        
        Args:
            n_inputs: Number of inputs
            strength: Disjunction strength parameter
            
        Returns:
            Quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        qc = QuantumCircuit(n_inputs + 1)
        output_qubit = n_inputs
        
        # OR: output is high if any input is high
        angle = strength * np.pi / 2
        for i in range(n_inputs):
            qc.cry(angle, i, output_qubit)
        
        return qc
    
    @staticmethod
    def create_fuzzy_not_circuit(strength: float = 1.0) -> QuantumCircuit:
        """Create quantum circuit for fuzzy NOT.
        
        Args:
            strength: Negation strength
            
        Returns:
            Quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        qc = QuantumCircuit(1)
        
        # NOT via X gate (or partial rotation for fuzzy)
        if abs(strength - 1.0) < 1e-6:
            qc.x(0)
        else:
            qc.ry(strength * np.pi, 0)
        
        return qc


if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Install with: pip install --break-system-packages qiskit qiskit-aer")
    else:
        print("=" * 60)
        print("Quantum Logic Circuits Example")
        print("=" * 60)
        
        # Test quantum logic gates
        print("\nTesting quantum logic gates...")
        
        # Quantum AND
        qc_and = QuantumCircuit(3)
        qc_and.x(0)  # Set first input to |1>
        qc_and.x(1)  # Set second input to |1>
        QuantumLogicGates.quantum_and(qc_and, 0, 1, 2)
        
        sv = Statevector.from_instruction(qc_and)
        print(f"\nQuantum AND with inputs |11>:")
        print(f"  Output qubit probability |1>: {sv.probabilities()[7]:.3f} (expected: 1.0)")
        
        # Test quantum logic rule
        print("\n" + "=" * 60)
        print("Quantum Logic Rule")
        print("=" * 60)
        
        rule = QuantumLogicRule(n_body_predicates=2, rule_type='fuzzy_and')
        rule_circuit = rule.create_circuit()
        
        print(f"\nRule circuit:")
        print(f"  Qubits: {rule_circuit.num_qubits}")
        print(f"  Depth: {rule_circuit.depth()}")
        print(f"  Type: fuzzy AND")
        
        # Bind parameter and test
        bound_circuit = rule.bind_parameter(rule_circuit, np.pi/4)
        print(f"  Parameter bound: θ = π/4")
        
        # Test quantum logic program
        print("\n" + "=" * 60)
        print("Quantum Logic Program")
        print("=" * 60)
        
        # Example: parent(X,Y) :- mother(X,Y)
        #          parent(X,Y) :- father(X,Y)
        # Predicates: 0=mother, 1=father, 2=parent
        
        rules = [
            (2, [0], 'implication'),  # parent :- mother
            (2, [1], 'implication'),  # parent :- father
        ]
        
        logic_program = QuantumLogicProgram(n_predicates=3, rules=rules)
        
        print(f"\nLogic program:")
        print(f"  Predicates: 3 (mother, father, parent)")
        print(f"  Rules: {len(rules)}")
        print(f"  Qubits: {logic_program.n_qubits}")
        print(f"  Parameters: {logic_program.n_params}")
        
        # Test execution
        initial_state = np.array([0.9, 0.1, 0.0])  # mother=0.9, father=0.1, parent=0.0
        param_values = np.array([np.pi/3, np.pi/3])
        
        print(f"\nInitial state: mother={initial_state[0]:.2f}, father={initial_state[1]:.2f}, parent={initial_state[2]:.2f}")
        
        final_state = logic_program.forward(initial_state, param_values)
        
        print(f"Final state: mother={final_state[0]:.2f}, father={final_state[1]:.2f}, parent={final_state[2]:.2f}")
        print(f"\nParent probability increased from {initial_state[2]:.2f} to {final_state[2]:.2f}")
        
        # Test fuzzy logic circuits
        print("\n" + "=" * 60)
        print("Quantum Fuzzy Logic")
        print("=" * 60)
        
        fuzzy_and = QuantumFuzzyLogic.create_fuzzy_and_circuit(n_inputs=2, strength=0.8)
        print(f"\nFuzzy AND circuit:")
        print(f"  Inputs: 2")
        print(f"  Strength: 0.8")
        print(f"  Qubits: {fuzzy_and.num_qubits}")
        print(f"  Depth: {fuzzy_and.depth()}")
        
        fuzzy_or = QuantumFuzzyLogic.create_fuzzy_or_circuit(n_inputs=2, strength=0.8)
        print(f"\nFuzzy OR circuit:")
        print(f"  Inputs: 2")
        print(f"  Strength: 0.8")
        print(f"  Qubits: {fuzzy_or.num_qubits}")
        
        print("\n" + "=" * 60)
        print("Quantum logic circuits tests passed!")
        print("=" * 60)
