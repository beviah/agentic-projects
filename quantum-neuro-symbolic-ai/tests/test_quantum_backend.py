"""Tests for QuantumBackend abstraction layer.

Mocks IBM Quantum Runtime SDK responses so all hardware paths
can be tested without credentials or network access.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from contextlib import nullcontext

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _bell_circuit():
    """Create a simple Bell state circuit (no measurements)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _measured_bell():
    """Bell circuit with measurements already attached."""
    qc = _bell_circuit()
    qc.measure_all()
    return qc


def _simple_observable():
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp.from_list([("ZZ", 1.0)])


# ---- Mock factories -------------------------------------------------------

def _mock_sampler_result(counts_list, shots=1024):
    """Build a fake SamplerV2 result object.

    Args:
        counts_list: list of dicts, one per circuit  e.g. [{"00": 512, "11": 512}]
        shots: reported shot count
    """
    pub_results = []
    for counts in counts_list:
        meas = MagicMock()
        meas.get_counts.return_value = counts
        data = MagicMock()
        data.meas = meas
        pub = MagicMock()
        pub.data = data
        pub_results.append(pub)

    result = MagicMock()
    result.__iter__ = lambda self: iter(pub_results)
    result.__getitem__ = lambda self, i: pub_results[i]
    return result


def _mock_estimator_result(evs_list):
    """Build a fake EstimatorV2 result object.

    Args:
        evs_list: list of floats, one expectation per PUB
    """
    pub_results = []
    for ev in evs_list:
        data = MagicMock()
        data.evs = np.array([ev])
        pub = MagicMock()
        pub.data = data
        pub_results.append(pub)

    result = MagicMock()
    result.__iter__ = lambda self: iter(pub_results)
    result.__getitem__ = lambda self, i: pub_results[i]
    return result


def _mock_runtime_service(backend_name="ibm_mock_backend"):
    """Return (service, backend) mocks that behave like QiskitRuntimeService."""
    backend = MagicMock()
    backend.name = backend_name
    # target needed by generate_preset_pass_manager — return None so we
    # patch the pass-manager separately
    backend.target = None

    service = MagicMock()
    service.least_busy.return_value = backend
    service.backend.return_value = backend
    return service, backend


# ---------------------------------------------------------------------------
# Tests — local simulator path  (no mocks needed, runs on AerSimulator)
# ---------------------------------------------------------------------------

class TestLocalBackend:
    """QuantumBackend with use_hardware=False (AerSimulator)."""

    def test_creation(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        config = QuantumBackendConfig(use_hardware=False, shots=512)
        backend = QuantumBackend(config)
        assert backend.is_hardware is False
        assert backend.config.shots == 512

    def test_run_sampler_adds_measurements(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False, shots=256))
        qc = _bell_circuit()                       # no measurements
        results = backend.run_sampler([qc], shots=256)
        assert len(results) == 1
        counts = results[0]["counts"]
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == 256

    def test_run_sampler_preserves_existing_measurements(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False))
        qc = _measured_bell()
        results = backend.run_sampler([qc])
        assert sum(results[0]["counts"].values()) == 1024

    def test_run_sampler_multiple_circuits(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False, shots=100))
        results = backend.run_sampler([_bell_circuit(), _bell_circuit()], shots=100)
        assert len(results) == 2

    def test_measure_probabilities(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False, shots=4096))
        probs = backend.measure_probabilities(_bell_circuit(), shots=4096)
        assert probs.shape == (4,)                 # 2 qubits → 4 outcomes
        assert abs(probs.sum() - 1.0) < 1e-6
        # Bell state: expect ~0.5 for |00⟩ and |11⟩
        assert probs[0] > 0.3
        assert probs[3] > 0.3

    def test_measure_expectation_z(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False, shots=2048))
        # |0⟩ state → ⟨Z⟩ = +1
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(1)                     # stays in |0⟩
        exp = backend.measure_expectation(qc, observable="Z", shots=2048)
        assert exp > 0.9                           # should be ~1.0

    def test_measure_expectation_x(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False, shots=2048))
        # H|0⟩ = |+⟩  → ⟨X⟩ = +1
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(1)
        qc.h(0)
        exp = backend.measure_expectation(qc, observable="X", shots=2048)
        assert exp > 0.9

    def test_run_estimator_local(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False))
        qc = _bell_circuit()
        obs = _simple_observable()
        expectations = backend.run_estimator([qc], [obs])
        assert len(expectations) == 1
        assert isinstance(expectations[0], (float, np.floating))

    def test_run_estimator_mismatched_lengths(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False))
        with pytest.raises(ValueError, match="must match"):
            backend.run_estimator([_bell_circuit()], [_simple_observable(), _simple_observable()])

    def test_create_session_local_yields_none(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False))
        with backend.create_session() as session:
            assert session is None

    def test_backend_name_property(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False))
        assert "aer" in backend.backend_name.lower() or "simulator" in backend.backend_name.lower()


# ---------------------------------------------------------------------------
# Tests — hardware path  (fully mocked, no credentials needed)
# ---------------------------------------------------------------------------

class TestHardwareBackend:
    """QuantumBackend with use_hardware=True, mocking IBM Runtime SDK."""

    # We need to patch:
    #   1. QiskitRuntimeService → returns our mock service
    #   2. generate_preset_pass_manager → returns identity (circuits pass through)
    #   3. SamplerV2 / EstimatorV2 → return canned results
    #   4. Session → context manager that does nothing

    @pytest.fixture(autouse=True)
    def _patch_runtime(self):
        """Patch IBM Runtime imports inside quantum_backend module."""
        service_mock, backend_mock = _mock_runtime_service()

        # Pass manager that returns circuits unchanged
        pm_mock = MagicMock()
        pm_mock.run = lambda circ: circ              # identity transpile

        patches = {
            "quantum_ml.quantum_backend.IBM_RUNTIME_AVAILABLE": True,
            "quantum_ml.quantum_backend.QiskitRuntimeService": MagicMock(return_value=service_mock),
            "quantum_ml.quantum_backend.generate_preset_pass_manager": MagicMock(return_value=pm_mock),
        }

        self._service = service_mock
        self._backend_mock = backend_mock
        self._pm = pm_mock

        with patch.dict("quantum_ml.quantum_backend.__dict__", patches):
            # Also need to patch at module attribute level for the class to pick up
            import quantum_ml.quantum_backend as mod
            orig_ibm = getattr(mod, "IBM_RUNTIME_AVAILABLE", False)
            orig_service = getattr(mod, "QiskitRuntimeService", None)
            orig_pm = getattr(mod, "generate_preset_pass_manager", None)

            mod.IBM_RUNTIME_AVAILABLE = True
            mod.QiskitRuntimeService = MagicMock(return_value=service_mock)
            mod.generate_preset_pass_manager = MagicMock(return_value=pm_mock)

            # Mock Session as context manager
            session_mock = MagicMock()
            session_mock.__enter__ = MagicMock(return_value=session_mock)
            session_mock.__exit__ = MagicMock(return_value=False)
            mod.Session = MagicMock(return_value=session_mock)
            self._session_cls = mod.Session

            yield

            mod.IBM_RUNTIME_AVAILABLE = orig_ibm
            if orig_service:
                mod.QiskitRuntimeService = orig_service
            if orig_pm:
                mod.generate_preset_pass_manager = orig_pm

    def _make_backend(self, **kw):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        config = QuantumBackendConfig(use_hardware=True, **kw)
        return QuantumBackend(config)

    # -- sampler tests -------------------------------------------------------

    def test_sampler_hardware_path(self):
        import quantum_ml.quantum_backend as mod
        sampler_instance = MagicMock()
        sampler_instance.options = MagicMock()
        sampler_instance.options.dynamical_decoupling = MagicMock()
        sampler_instance.run.return_value.result.return_value = _mock_sampler_result(
            [{"00": 500, "11": 524}], shots=1024
        )
        mod.Sampler = MagicMock(return_value=sampler_instance)

        backend = self._make_backend(shots=1024)
        results = backend.run_sampler([_bell_circuit()], shots=1024)

        assert len(results) == 1
        assert results[0]["counts"] == {"00": 500, "11": 524}
        sampler_instance.run.assert_called_once()

    def test_sampler_dynamical_decoupling(self):
        import quantum_ml.quantum_backend as mod
        sampler_instance = MagicMock()
        sampler_instance.options = MagicMock()
        sampler_instance.options.dynamical_decoupling = MagicMock()
        sampler_instance.run.return_value.result.return_value = _mock_sampler_result(
            [{"00": 512, "11": 512}]
        )
        mod.Sampler = MagicMock(return_value=sampler_instance)

        backend = self._make_backend(enable_dynamical_decoupling=True)
        backend.run_sampler([_bell_circuit()])

        # Verify DD was enabled on options
        assert sampler_instance.options.dynamical_decoupling.enable is True

    # -- estimator tests -----------------------------------------------------

    def test_estimator_hardware_path(self):
        import quantum_ml.quantum_backend as mod
        estimator_instance = MagicMock()
        estimator_instance.options = MagicMock()
        estimator_instance.options.dynamical_decoupling = MagicMock()
        estimator_instance.run.return_value.result.return_value = _mock_estimator_result([0.75])
        mod.Estimator = MagicMock(return_value=estimator_instance)

        backend = self._make_backend()

        # Need circuit with layout attr for apply_layout mock
        qc = _bell_circuit()
        obs = _simple_observable()
        # Mock apply_layout since our "transpiled" circuit has no real layout
        obs.apply_layout = MagicMock(return_value=obs)

        expectations = backend.run_estimator([qc], [obs])
        assert len(expectations) == 1
        assert abs(expectations[0] - 0.75) < 1e-9

    def test_estimator_resilience_level(self):
        import quantum_ml.quantum_backend as mod
        estimator_instance = MagicMock()
        estimator_instance.options = MagicMock()
        estimator_instance.options.dynamical_decoupling = MagicMock()
        estimator_instance.run.return_value.result.return_value = _mock_estimator_result([0.5])
        mod.Estimator = MagicMock(return_value=estimator_instance)

        backend = self._make_backend(resilience_level=2)

        qc = _bell_circuit()
        obs = _simple_observable()
        obs.apply_layout = MagicMock(return_value=obs)
        backend.run_estimator([qc], [obs])

        assert estimator_instance.options.resilience_level == 2

    # -- session tests -------------------------------------------------------

    def test_create_session_hardware(self):
        backend = self._make_backend()
        with backend.create_session() as session:
            assert session is not None
        self._session_cls.assert_called_once()

    # -- property tests ------------------------------------------------------

    def test_is_hardware_true(self):
        backend = self._make_backend()
        assert backend.is_hardware is True


# ---------------------------------------------------------------------------
# Tests — VQE integration
# ---------------------------------------------------------------------------

class TestVQEBackend:
    """VQE with local QuantumBackend (no hardware mocks needed)."""

    def test_vqe_without_backend(self):
        from quantum_ml.variational_circuits import VQE
        from qiskit.quantum_info import SparsePauliOp
        H = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 1.0)])
        vqe = VQE(H, ansatz_type="hardware_efficient", n_qubits=2, depth=1)
        assert vqe.backend is None
        # Single evaluation should work
        params = np.random.randn(len(vqe.params)) * 0.1
        e = vqe.expectation_value(params)
        assert isinstance(e, (float, np.floating))

    def test_vqe_with_local_backend(self):
        from quantum_ml.variational_circuits import VQE
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        from qiskit.quantum_info import SparsePauliOp

        H = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 1.0)])
        config = QuantumBackendConfig(use_hardware=False)
        backend = QuantumBackend(config)
        vqe = VQE(H, ansatz_type="hardware_efficient", n_qubits=2, depth=1, backend=backend)
        assert vqe.backend is not None

        params = np.random.randn(len(vqe.params)) * 0.1
        e = vqe.expectation_value(params)
        assert isinstance(e, (float, np.floating))

    def test_vqe_run_converges(self):
        from quantum_ml.variational_circuits import VQE
        from qiskit.quantum_info import SparsePauliOp
        # H = Z⊗I + I⊗Z  → ground energy = -2
        H = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 1.0)])
        vqe = VQE(H, ansatz_type="hardware_efficient", n_qubits=2, depth=1)
        result = vqe.run(maxiter=30)
        assert result["optimal_energy"] < result["history"][0]
        assert len(result["history"]) > 1


# ---------------------------------------------------------------------------
# Tests — QuantumKernel integration
# ---------------------------------------------------------------------------

class TestQuantumKernel:

    def test_kernel_self_fidelity(self):
        from quantum_ml.quantum_kernels import QuantumKernel
        kernel = QuantumKernel(feature_map="zz", n_qubits=2, depth=1, method="fidelity")
        x = np.array([0.5, 0.3])
        k = kernel.kernel_function(x, x)
        assert abs(k - 1.0) < 1e-4, f"Self-kernel should be ~1.0, got {k}"

    def test_kernel_different_points(self):
        from quantum_ml.quantum_kernels import QuantumKernel
        kernel = QuantumKernel(feature_map="zz", n_qubits=2, depth=1, method="fidelity")
        x1 = np.array([0.0, 0.0])
        x2 = np.array([np.pi, np.pi])
        k = kernel.kernel_function(x1, x2)
        assert 0.0 <= k <= 1.0

    def test_kernel_matrix_shape(self):
        from quantum_ml.quantum_kernels import QuantumKernel
        kernel = QuantumKernel(feature_map="zz", n_qubits=2, depth=1)
        X = np.random.rand(3, 2)
        K = kernel.compute_kernel_matrix(X)
        assert K.shape == (3, 3)
        # Diagonal should be ~1
        for i in range(3):
            assert K[i, i] > 0.9

    def test_kernel_with_backend(self):
        from quantum_ml.quantum_kernels import QuantumKernel
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        config = QuantumBackendConfig(use_hardware=False, shots=512)
        backend = QuantumBackend(config)
        kernel = QuantumKernel(feature_map="zz", n_qubits=2, depth=1,
                               method="fidelity", backend=backend)
        x = np.array([0.5, 0.3])
        k = kernel.kernel_function(x, x)
        # SWAP-test based — noisier, but should still be high for identical inputs
        assert k > 0.5


# ---------------------------------------------------------------------------
# Tests — QSVM
# ---------------------------------------------------------------------------

class TestQuantumSVM:

    def test_qsvm_fit_predict(self):
        from quantum_ml.quantum_kernels import QuantumKernel, QuantumSVM
        kernel = QuantumKernel(feature_map="zz", n_qubits=2, depth=1)
        X_train = np.array([[0.1, 0.1], [0.2, 0.2], [0.8, 0.8], [0.9, 0.9]])
        y_train = np.array([-1, -1, 1, 1])
        svm = QuantumSVM(kernel, C=1.0)
        svm.fit(X_train, y_train)
        preds = svm.predict(X_train)
        assert preds.shape == (4,)
        accuracy = svm.score(X_train, y_train)
        assert accuracy >= 0.5, f"Training accuracy too low: {accuracy}"

    def test_qsvm_score_range(self):
        from quantum_ml.quantum_kernels import QuantumKernel, QuantumSVM
        kernel = QuantumKernel(feature_map="zz", n_qubits=2, depth=1)
        X = np.array([[0.1, 0.1], [0.9, 0.9]])
        y = np.array([-1, 1])
        svm = QuantumSVM(kernel, C=1.0)
        svm.fit(X, y)
        assert 0.0 <= svm.score(X, y) <= 1.0


# ---------------------------------------------------------------------------
# Tests — Neuro-symbolic modules accept backend param
# ---------------------------------------------------------------------------

class TestNeuroSymbolicBackendParam:
    """Verify all quantum neuro-symbolic modules accept and store backend."""

    def _local_backend(self):
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        return QuantumBackend(QuantumBackendConfig(use_hardware=False))

    def test_quantum_logic_program(self):
        from quantum_neuro_symbolic.quantum_logic_circuits import QuantumLogicProgram
        rules = [(2, [0], "implication"), (2, [1], "implication")]
        prog = QuantumLogicProgram(n_predicates=3, rules=rules, backend=self._local_backend())
        assert prog.backend is not None

    def test_quantum_logic_program_forward(self):
        from quantum_neuro_symbolic.quantum_logic_circuits import QuantumLogicProgram
        rules = [(2, [0], "implication"), (2, [1], "implication")]
        prog = QuantumLogicProgram(n_predicates=3, rules=rules)
        state = np.array([0.9, 0.1, 0.0])
        params = np.array([np.pi / 3, np.pi / 3])
        result = prog.forward(state, params)
        assert result.shape == (3,)
        # parent probability should have increased
        assert result[2] > state[2]

    def test_quantum_kg_embedding(self):
        from quantum_neuro_symbolic.quantum_kg_embedding import QuantumKGEmbedding
        kg = QuantumKGEmbedding(n_entities=4, n_relations=2, n_qubits=3,
                                backend=self._local_backend())
        assert kg.backend is not None

    def test_quantum_kg_triple_score(self):
        from quantum_neuro_symbolic.quantum_kg_embedding import QuantumKGEmbedding
        # Use n_qubits>=3 to avoid edge cases with transitive relation param count
        kg = QuantumKGEmbedding(n_entities=4, n_relations=2, n_qubits=3)
        score = kg.compute_triple_score(0, 0, 1)
        assert 0.0 <= score <= 1.0

    def test_quantum_cbm(self):
        import torch
        from quantum_neuro_symbolic.quantum_cbm import QuantumCBM
        cbm = QuantumCBM(input_dim=4, n_concepts=2, n_classes=2, n_qubits=4,
                         backend=self._local_backend())
        assert cbm.backend is not None

    def test_quantum_cbm_forward(self):
        import torch
        from quantum_neuro_symbolic.quantum_cbm import QuantumCBM
        cbm = QuantumCBM(input_dim=4, n_concepts=2, n_classes=2, n_qubits=4)
        x = torch.randn(1, 4)
        logits, concepts = cbm(x)
        assert logits.shape == (1, 2)
        assert concepts.shape == (1, 2)

    def test_quantum_gnn(self):
        from quantum_neuro_symbolic.quantum_gnn import QuantumGNN
        gnn = QuantumGNN(n_nodes=3, input_dim=4, n_qubits=2, n_layers=1,
                         backend=self._local_backend())
        assert gnn.backend is not None

    def test_quantum_gnn_forward(self):
        import torch
        from quantum_neuro_symbolic.quantum_gnn import QuantumGNN
        gnn = QuantumGNN(n_nodes=3, input_dim=4, n_qubits=2, n_layers=1, hidden_dim=8)
        features = torch.randn(3, 4)
        adj = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        out = gnn(features, adj)
        assert out.shape == (3, 8)


# ---------------------------------------------------------------------------
# Tests — Classical neuro-symbolic (no quantum deps)
# ---------------------------------------------------------------------------

class TestClassicalNeuroSymbolic:

    def test_differentiable_logic(self):
        import torch
        from neuro_symbolic.differentiable_logic import DifferentiableLogicProgram
        rules = [(2, [0]), (2, [1]), (3, [2])]
        prog = DifferentiableLogicProgram(n_predicates=4, rules=rules)
        facts = torch.tensor([[0.9, 0.0, 0.0, 0.0]])
        derived = prog(facts)
        assert derived.shape == (1, 4)
        assert derived[0, 2].item() > 0.5   # parent derived from mother

    def test_concept_bottleneck(self):
        import torch
        from neuro_symbolic.concept_bottleneck import IndependentCBM
        cbm = IndependentCBM(input_dim=8, n_concepts=3, n_classes=2)
        x = torch.randn(2, 8)
        concepts, logits = cbm(x)
        assert concepts.shape == (2, 3)
        assert logits.shape == (2, 2)
        # concepts should be in [0, 1] (sigmoid)
        assert (concepts >= 0).all() and (concepts <= 1).all()

    def test_concept_intervention(self):
        import torch
        from neuro_symbolic.concept_bottleneck import IndependentCBM
        cbm = IndependentCBM(input_dim=8, n_concepts=3, n_classes=2)
        x = torch.randn(1, 8)
        intervention = torch.tensor([[-1.0, 1.0, -1.0]])  # force concept 1 = 1.0
        concepts, logits = cbm(x, intervene_concepts=intervention)
        assert abs(concepts[0, 1].item() - 1.0) < 1e-6

    def test_knowledge_guided_gnn(self):
        import torch
        from neuro_symbolic.knowledge_guided_nn import KnowledgeGraph, KGGuidedGNN
        kg = KnowledgeGraph(n_entities=4, n_relations=1)
        kg.add_triple(0, 0, 1)
        kg.add_triple(1, 0, 2)
        gnn = KGGuidedGNN(n_entities=4, n_relations=1, embedding_dim=8, hidden_dim=8)
        emb = gnn(kg)
        assert emb.shape == (4, 8)


# ---------------------------------------------------------------------------
# Tests — Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:

    def test_dataset_shapes(self):
        from benchmarks.simple_benchmark import SyntheticRelationalDataset
        ds = SyntheticRelationalDataset(n_samples=50, n_features=6, seed=0)
        assert ds.X.shape == (50, 6)
        assert ds.y.shape == (50,)

    def test_dataset_labels_valid(self):
        import torch
        from benchmarks.simple_benchmark import SyntheticRelationalDataset
        ds = SyntheticRelationalDataset(n_samples=100, seed=0)
        assert set(ds.y.tolist()).issubset({0, 1, 2})

    def test_splits(self):
        from benchmarks.simple_benchmark import SyntheticRelationalDataset
        ds = SyntheticRelationalDataset(n_samples=100, seed=0)
        Xtr, ytr, Xte, yte = ds.get_splits(0.8)
        assert Xtr.shape[0] == 80
        assert Xte.shape[0] == 20

    def test_ffn_trains(self):
        import torch
        from benchmarks.simple_benchmark import (
            SyntheticRelationalDataset, SimpleFFN, train_model
        )
        torch.manual_seed(0)
        ds = SyntheticRelationalDataset(n_samples=60, seed=0)
        Xtr, ytr, _, _ = ds.get_splits(0.8)
        model = SimpleFFN(input_dim=10, n_classes=3)
        history = train_model(model, Xtr, ytr, epochs=10, verbose=False)
        assert history["loss"][-1] < history["loss"][0]


# ---------------------------------------------------------------------------
# Tests — Edge cases & error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_circuit(self):
        """Backend should handle a circuit with no gates."""
        from qiskit import QuantumCircuit
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        qc = QuantumCircuit(1)
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False, shots=100))
        results = backend.run_sampler([qc], shots=100)
        counts = results[0]["counts"]
        # Should measure |0⟩ with probability 1
        assert counts.get("0", 0) == 100

    def test_single_qubit_probabilities(self):
        from qiskit import QuantumCircuit
        from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig
        qc = QuantumCircuit(1)
        qc.x(0)   # flip to |1⟩
        backend = QuantumBackend(QuantumBackendConfig(use_hardware=False, shots=500))
        probs = backend.measure_probabilities(qc, shots=500)
        assert probs.shape == (2,)
        assert probs[1] > 0.95

    def test_quantum_kernel_swap_test(self):
        """Ensure swap_test method doesn't crash."""
        from quantum_ml.quantum_kernels import QuantumKernel
        kernel = QuantumKernel(feature_map="zz", n_qubits=2, depth=1, method="swap_test")
        x = np.array([0.5, 0.3])
        k = kernel.kernel_function(x, x)
        assert 0.0 <= k <= 1.5   # swap test can be slightly > 1 due to shot noise

    def test_qsvm_predict_before_fit_raises(self):
        from quantum_ml.quantum_kernels import QuantumKernel, QuantumSVM
        kernel = QuantumKernel(feature_map="zz", n_qubits=2, depth=1)
        svm = QuantumSVM(kernel)
        with pytest.raises(ValueError, match="not trained"):
            svm.predict(np.array([[0.1, 0.1]]))


# ---------------------------------------------------------------------------
# conftest.py helper — add project root to sys.path
# ---------------------------------------------------------------------------
# NOTE: Create tests/conftest.py with:
#
#   import sys, os
#   sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#
# Then run:  python -m pytest tests/ -v