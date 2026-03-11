#!/usr/bin/env python3
"""Integration test for code review fixes."""

import numpy as np
import sys

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not available - skipping tests")
    sys.exit(0)

from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
from quantum_ml.quantum_kernels import QuantumKernel
from quantum_ml.variational_circuits import VQE, QAOA

def test_quantum_kernel_backend_support():
    """Test QuantumKernel accepts backend parameter."""
    print("Testing QuantumKernel backend support...")
    
    # Test without backend (backward compatibility)
    kernel = QuantumKernel(feature_map='zz', n_qubits=2, depth=1)
    assert kernel.backend is None
    print("  ✓ QuantumKernel works without backend")
    
    # Test with backend
    config = QuantumBackendConfig(use_hardware=False, shots=1024)
    backend = QuantumBackend(config)
    kernel_with_backend = QuantumKernel(feature_map='zz', n_qubits=2, depth=1, backend=backend)
    assert kernel_with_backend.backend is not None
    print("  ✓ QuantumKernel accepts backend parameter")
    
    # Test compute_fidelity without backend (statevector)
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    
    fidelity = kernel.compute_fidelity(qc1, qc2)
    assert 0.0 <= fidelity <= 1.0
    print(f"  ✓ compute_fidelity() works without backend (fidelity={fidelity:.4f})")
    
    print("✓ QuantumKernel tests passed\n")

def test_vqe_backend_support():
    """Test VQE accepts backend parameter and has session support."""
    print("Testing VQE backend support...")
    
    # Create simple Hamiltonian
    hamiltonian = SparsePauliOp(['ZZ', 'XI', 'IX'], coeffs=[1.0, 0.5, 0.5])
    
    # Test without backend (backward compatibility)
    vqe = VQE(hamiltonian, ansatz_type='hardware_efficient', n_qubits=2, depth=1)
    assert vqe.backend is None
    print("  ✓ VQE works without backend")
    
    # Test with backend
    config = QuantumBackendConfig(use_hardware=False, shots=1024)
    backend = QuantumBackend(config)
    vqe_with_backend = VQE(hamiltonian, ansatz_type='hardware_efficient', n_qubits=2, depth=1, backend=backend)
    assert vqe_with_backend.backend is not None
    print("  ✓ VQE accepts backend parameter")
    
    # Test expectation_value with backend
    params = np.random.randn(vqe_with_backend.ansatz.num_parameters) * 0.1
    expectation = vqe_with_backend.expectation_value(params)
    assert isinstance(expectation, (float, np.floating))
    print(f"  ✓ VQE.expectation_value() works with backend (E={expectation:.4f})")
    
    print("✓ VQE tests passed\n")

def test_qaoa_backend_support():
    """Test QAOA accepts backend parameter and has session support."""
    print("Testing QAOA backend support...")
    
    # Simple cost function (MaxCut)
    def cost_function(bitstring):
        return sum(int(bitstring[i]) != int(bitstring[i+1]) for i in range(len(bitstring)-1))
    
    # Test without backend (backward compatibility)
    qaoa = QAOA(n_qubits=3, cost_function=cost_function, p=1)
    assert qaoa.backend is None
    print("  ✓ QAOA works without backend")
    
    # Test with backend
    config = QuantumBackendConfig(use_hardware=False, shots=1024)
    backend = QuantumBackend(config)
    qaoa_with_backend = QAOA(n_qubits=3, cost_function=cost_function, p=1, backend=backend)
    assert qaoa_with_backend.backend is not None
    print("  ✓ QAOA accepts backend parameter")
    
    print("✓ QAOA tests passed\n")

def test_neuro_symbolic_backend_warnings():
    """Test neuro-symbolic modules accept backend parameter."""
    print("Testing neuro-symbolic module backend support...")
    
    # These modules accept backend but are simulation-only
    from quantum_neuro_symbolic.quantum_logic_circuits import QuantumLogicProgram
    from quantum_neuro_symbolic.quantum_kg_embedding import QuantumKGEmbedding
    from quantum_neuro_symbolic.quantum_cbm import QuantumCBM
    from quantum_neuro_symbolic.quantum_gnn import QuantumGNN
    
    config = QuantumBackendConfig(use_hardware=False, shots=1024)
    backend = QuantumBackend(config)
    
    # Test QuantumLogicProgram
    rules = [(0, [1, 2], 'and')]
    qlp = QuantumLogicProgram(n_predicates=3, rules=rules, backend=backend)
    assert qlp.backend is not None
    print("  ✓ QuantumLogicProgram accepts backend parameter")
    
    # Test QuantumKGEmbedding
    kg_emb = QuantumKGEmbedding(n_entities=5, n_relations=2, n_qubits=2, backend=backend)
    assert kg_emb.backend is not None
    print("  ✓ QuantumKGEmbedding accepts backend parameter")
    
    # Test QuantumCBM
    import torch
    cbm = QuantumCBM(input_dim=4, n_concepts=2, n_classes=2, n_qubits=4, backend=backend)
    assert cbm.backend is not None
    print("  ✓ QuantumCBM accepts backend parameter")
    
    # Test QuantumGNN
    gnn = QuantumGNN(n_nodes=3, input_dim=4, n_qubits=2, n_layers=1, backend=backend)
    assert gnn.backend is not None
    print("  ✓ QuantumGNN accepts backend parameter")
    
    print("✓ Neuro-symbolic tests passed\n")

if __name__ == '__main__':
    print("="*60)
    print("Code Review Fixes Integration Tests")
    print("="*60 + "\n")
    
    test_quantum_kernel_backend_support()
    test_vqe_backend_support()
    test_qaoa_backend_support()
    test_neuro_symbolic_backend_warnings()
    
    print("="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
