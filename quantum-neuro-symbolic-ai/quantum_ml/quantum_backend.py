"""Quantum Backend Abstraction for IBM Quantum Runtime.

Provides unified interface for executing quantum circuits on both local simulators
and IBM Quantum hardware using the Qiskit Runtime primitives.
"""

from __future__ import annotations

import os
import numpy as np
from typing import List, Optional, Union, Dict, Any
from contextlib import contextmanager

try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Install with: pip install qiskit qiskit-aer")

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False
    Session = None
    Sampler = None
    Estimator = None


class QuantumBackendConfig:
    """Configuration for quantum backend."""
    
    def __init__(self,
                 use_hardware: bool = False,
                 backend_name: Optional[str] = None,
                 optimization_level: int = 1,
                 resilience_level: int = 0,
                 shots: int = 1024,
                 enable_dynamical_decoupling: bool = False):
        """Initialize backend configuration.
        
        Args:
            use_hardware: If True, use IBM Quantum hardware; if False, use local simulator
            backend_name: Specific backend name (None = least busy)
            optimization_level: Transpiler optimization level (0-3)
            resilience_level: Error mitigation level (0-2)
            shots: Default number of measurement shots
            enable_dynamical_decoupling: Enable dynamical decoupling for error mitigation
        """
        self.use_hardware = use_hardware
        self.backend_name = backend_name
        self.optimization_level = optimization_level
        self.resilience_level = resilience_level
        self.shots = shots
        self.enable_dynamical_decoupling = enable_dynamical_decoupling


class QuantumBackend:
    """Unified quantum backend for local simulation and IBM Quantum hardware.
    
    This class provides a consistent interface for executing quantum circuits
    on either local simulators (AerSimulator) or IBM Quantum hardware using
    the Qiskit Runtime primitives (Sampler and Estimator).
    
    Examples:
        >>> # Local simulation
        >>> backend = QuantumBackend(use_hardware=False)
        >>> result = backend.run_sampler([circuit], shots=1024)
        
        >>> # IBM Quantum hardware
        >>> backend = QuantumBackend(use_hardware=True)
        >>> result = backend.run_estimator([circuit], [observable])
    """
    
    def __init__(self, config: Optional[QuantumBackendConfig] = None):
        """Initialize quantum backend.
        
        Args:
            config: Backend configuration (uses defaults if None)
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required. Install with: pip install qiskit qiskit-aer")
        
        self.config = config or QuantumBackendConfig()
        self._service = None
        self._backend = None
        self._pass_manager = None
        self._session = None
        self._active_session = None  # Track active session for primitives
        
        # Initialize backend
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the quantum backend (simulator or hardware)."""
        if self.config.use_hardware:
            if not IBM_RUNTIME_AVAILABLE:
                raise ImportError(
                    "qiskit-ibm-runtime required for hardware access. "
                    "Install with: pip install qiskit-ibm-runtime"
                )
            
            # Initialize IBM Quantum Runtime service
            self._service = QiskitRuntimeService()
            
            # Select backend
            if self.config.backend_name:
                self._backend = self._service.backend(self.config.backend_name)
            else:
                # Use least busy backend
                self._backend = self._service.least_busy(operational=True, simulator=False)
            
            print(f"Using IBM Quantum backend: {self._backend.name}")
        else:
            # Use local AerSimulator
            self._backend = AerSimulator()
            print("Using local AerSimulator")
        
        # Create transpiler pass manager
        self._pass_manager = generate_preset_pass_manager(
            backend=self._backend,
            optimization_level=self.config.optimization_level
        )
    
    def _transpile_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile circuit to ISA format for target backend.
        
        Args:
            circuit: Input quantum circuit
            
        Returns:
            Transpiled ISA circuit
        """
        return self._pass_manager.run(circuit)
    
    def run_sampler(self, 
                   circuits: Union[QuantumCircuit, List[QuantumCircuit]],
                   shots: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute circuits and return measurement counts.
        
        Args:
            circuits: Circuit or list of circuits to execute
            shots: Number of measurement shots (uses config default if None)
            
        Returns:
            List of result dictionaries with 'counts' key
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        
        shots = shots or self.config.shots
        
        # Ensure circuits have measurements
        measured_circuits = []
        for circuit in circuits:
            if circuit.num_clbits == 0:
                # Add measurement if not present
                measured_circuit = circuit.copy()
                measured_circuit.measure_all()
                measured_circuits.append(measured_circuit)
            else:
                measured_circuits.append(circuit)
        
        if self.config.use_hardware:
            # IBM Quantum Runtime execution
            isa_circuits = [self._transpile_circuit(circ) for circ in measured_circuits]
            
            # Use active session if available, otherwise use backend directly
            mode = self._active_session if self._active_session is not None else self._backend
            sampler = Sampler(mode=mode)
            sampler.options.default_shots = shots
            
            # Configure error mitigation
            if self.config.enable_dynamical_decoupling:
                sampler.options.dynamical_decoupling.enable = True
            
            # Execute
            job = sampler.run(isa_circuits)
            result = job.result()
            
            # Extract counts from each circuit result
            results = []
            for i, pub_result in enumerate(result):
                counts = pub_result.data.meas.get_counts()
                results.append({'counts': counts, 'shots': shots})
            
            return results
        else:
            # Local simulator execution
            results = []
            for circuit in measured_circuits:
                sim_result = self._backend.run(circuit, shots=shots).result()
                counts = sim_result.get_counts()
                results.append({'counts': counts, 'shots': shots})
            
            return results
    
    def run_estimator(self,
                     circuits: Union[QuantumCircuit, List[QuantumCircuit]],
                     observables: Union[SparsePauliOp, List[SparsePauliOp]]) -> List[float]:
        """Execute circuits and return expectation values of observables.
        
        Args:
            circuits: Circuit or list of circuits
            observables: Observable or list of observables
            
        Returns:
            List of expectation values
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        if isinstance(observables, SparsePauliOp):
            observables = [observables]
        
        if len(circuits) != len(observables):
            raise ValueError("Number of circuits must match number of observables")
        
        if self.config.use_hardware:
            # IBM Quantum Runtime execution
            isa_circuits = [self._transpile_circuit(circ) for circ in circuits]
            
            # Apply layout to observables
            isa_observables = [
                obs.apply_layout(isa_circ.layout)
                for obs, isa_circ in zip(observables, isa_circuits)
            ]
            
            # Use active session if available, otherwise use backend directly
            mode = self._active_session if self._active_session is not None else self._backend
            estimator = Estimator(mode=mode)
            
            # Configure error mitigation
            if self.config.resilience_level > 0:
                estimator.options.resilience_level = self.config.resilience_level
            if self.config.enable_dynamical_decoupling:
                estimator.options.dynamical_decoupling.enable = True
            
            # Execute
            pubs = list(zip(isa_circuits, isa_observables))
            job = estimator.run(pubs)
            result = job.result()
            
            # Extract expectation values (evs may be scalar or array)
            expectations = []
            for pub_result in result:
                evs = pub_result.data.evs
                val = float(evs) if np.ndim(evs) == 0 else float(evs[0])
                expectations.append(val)
            return expectations
        else:
            # Local simulator using statevector
            from qiskit.quantum_info import Statevector
            
            expectations = []
            for circuit, observable in zip(circuits, observables):
                # Strip measurements if present (Statevector can't handle them)
                if circuit.num_clbits > 0:
                    clean_circuit = circuit.remove_final_measurements(inplace=False)
                else:
                    clean_circuit = circuit
                # Get statevector
                statevector = Statevector.from_instruction(clean_circuit)
                # Compute expectation
                expectation = statevector.expectation_value(observable).real
                expectations.append(expectation)
            
            return expectations
            
    @contextmanager
    def create_session(self):
        """Create a session for iterative quantum algorithms.
        
        Sessions allow multiple quantum jobs to be prioritized together,
        which is useful for variational algorithms like VQE and QAOA.
        
        Yields:
            Session object (or None for local simulation)
        
        Example:
            >>> with backend.create_session() as session:
            ...     for iteration in range(10):
            ...         result = backend.run_estimator([circuit], [observable])
        """
        if self.config.use_hardware:
            if not IBM_RUNTIME_AVAILABLE:
                raise ImportError("qiskit-ibm-runtime required for sessions")
            
            session = Session(backend=self._backend)
            self._active_session = session
            try:
                yield session
            finally:
                self._active_session = None
                session.close()
        else:
            # No session needed for local simulation
            yield None
    
    def measure_probabilities(self, circuit: QuantumCircuit, shots: int = 1024) -> np.ndarray:
        """Measure probability distribution from circuit.
        
        Args:
            circuit: Quantum circuit
            shots: Number of measurement shots
            
        Returns:
            Probability array of shape (2^n_qubits,)
        """
        results = self.run_sampler([circuit], shots=shots)
        counts = results[0]['counts']
        
        # Convert counts to probability array
        n_outcomes = 2 ** circuit.num_qubits
        probs = np.zeros(n_outcomes)
        
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            probs[idx] = count / shots
        
        return probs
    
    def measure_expectation(self, circuit: QuantumCircuit, 
                          observable: str = 'Z', shots: int = 1000) -> float:
        """Measure expectation value of an observable.
        
        Args:
            circuit: Quantum circuit to measure
            observable: Observable to measure ('Z', 'X', 'Y')
            shots: Number of measurement shots
            
        Returns:
            Expectation value
        """
        # Add measurement basis rotation if needed
        measured_circuit = circuit.copy()
        if observable == 'X':
            for qubit in range(circuit.num_qubits):
                measured_circuit.h(qubit)
        elif observable == 'Y':
            for qubit in range(circuit.num_qubits):
                measured_circuit.sdg(qubit)
                measured_circuit.h(qubit)
        
        # Measure
        results = self.run_sampler([measured_circuit], shots=shots)
        counts = results[0]['counts']
        
        # Calculate expectation value
        expectation = 0.0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Count number of 1s (assume Z basis)
            num_ones = bitstring.count('1')
            parity = (-1) ** num_ones
            expectation += parity * count / total_shots
        
        return expectation
    
    @property
    def backend_name(self) -> str:
        """Get the name of the current backend."""
        return self._backend.name if hasattr(self._backend, 'name') else 'AerSimulator'
    
    @property
    def is_hardware(self) -> bool:
        """Check if using real quantum hardware."""
        return self.config.use_hardware


def setup_ibm_quantum(token: str, channel: str = "ibm_quantum", overwrite: bool = False):
    """Setup IBM Quantum credentials.
    
    Args:
        token: IBM Quantum API token
        channel: Channel type ('ibm_quantum' or 'ibm_cloud')
        overwrite: If True, overwrite existing credentials
    
    Example:
        >>> setup_ibm_quantum(token="YOUR_API_TOKEN")
        IBM Quantum credentials saved successfully!
    """
    if not IBM_RUNTIME_AVAILABLE:
        raise ImportError(
            "qiskit-ibm-runtime required. "
            "Install with: pip install qiskit-ibm-runtime"
        )
    
    QiskitRuntimeService.save_account(
        channel=channel,
        token=token,
        overwrite=overwrite
    )
    print("IBM Quantum credentials saved successfully!")
    print(f"Credentials stored in: {os.path.expanduser('~/.qiskit/qiskit-ibm.json')}")


if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Install with: pip install --break-system-packages qiskit qiskit-aer")
    else:
        print("=" * 60)
        print("Quantum Backend Abstraction Example")
        print("=" * 60)
        
        # Create test circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        print("\nTest Circuit:")
        print(qc.draw(output='text'))
        
        # Test local backend
        print("\n" + "=" * 60)
        print("Local Simulator Backend")
        print("=" * 60)
        
        config = QuantumBackendConfig(use_hardware=False, shots=1024)
        backend = QuantumBackend(config)
        
        print(f"\nBackend: {backend.backend_name}")
        print(f"Is hardware: {backend.is_hardware}")
        
        # Test sampler
        print("\nRunning sampler...")
        results = backend.run_sampler([qc], shots=1024)
        print(f"Counts: {results[0]['counts']}")
        
        # Test probability measurement
        probs = backend.measure_probabilities(qc, shots=1024)
        print(f"\nProbabilities: {probs}")
        print(f"P(|00>): {probs[0]:.3f}")
        print(f"P(|11>): {probs[3]:.3f}")
        
        # Test expectation measurement
        expectation = backend.measure_expectation(qc, observable='Z', shots=1000)
        print(f"\nExpectation <Z>: {expectation:.3f}")
        
        print("\n" + "=" * 60)
        print("Quantum backend tests passed!")
        print("=" * 60)
        
        if IBM_RUNTIME_AVAILABLE:
            print("\n" + "=" * 60)
            print("IBM Quantum Runtime Available")
            print("=" * 60)
            print("\nTo use IBM Quantum hardware:")
            print("1. Get your API token from: https://quantum.ibm.com/")
            print("2. Run: setup_ibm_quantum(token='YOUR_TOKEN')")
            print("3. Create backend with: QuantumBackend(QuantumBackendConfig(use_hardware=True))")
        else:
            print("\n" + "=" * 60)
            print("IBM Quantum Runtime Not Available")
            print("=" * 60)
            print("\nTo enable IBM Quantum hardware access:")
            print("pip install --break-system-packages qiskit-ibm-runtime")
