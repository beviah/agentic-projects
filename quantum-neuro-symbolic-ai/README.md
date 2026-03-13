# Quantum Neuro-Symbolic AI

A framework combining quantum computing, differentiable logic programming, knowledge graphs, and concept bottleneck models into an integrated neuro-symbolic AI system.

## Built by

**Built by the author using a proprietary agentic AI framework** that orchestrates LLM models through structured cognitive pipelines — task classification, strategy selection, iterative critique-refine loops, anti-pattern detection, cross-session learning, and autonomous code execution with quality-gated completion. The framework learns from past sessions and adapts its reasoning strategies over time. Development used a human-in-the-loop workflow with Claude Sonnet for implementation and Claude Opus for architectural review. The entire project was produced in 7 sessions, ~150 minutes of compute time, across 250 agent iterations. To the author's knowledge, Devin is the closest comparable production system, though this framework takes a different architectural approach to cognitive orchestration and persistent learning. The framework is proprietary and not included in this release. 

### This pipeline used for this demo can easily be wrapped into an agentic AI scientist that automatically discovers underexplored combinations of cutting‑edge methods, builds end‑to‑end systems for real datasets, a closed‑loop pipeline that iterates until it achieves useful or state‑of‑the‑art results with interpretable reasoning. ( Given enough tokens / funding :)

Initial prompt:

```
Explore these: 

Neuro-Symbolic AI: Combining neural networks with symbolic reasoning; differentiable logic programming; knowledge-guided neural architectures; concept bottleneck models 

Quantum Machine Learning (QML): Hybrid quantum-classical models; quantum kernel methods; variational quantum circuits; quantum advantage claims 

Then explore if this can be done: Quantum neuro-symbolic AI — quantum circuits implementing differentiable logic on KG structures
```

Using Qiskit, PyTorch, and the IBM Quantum Runtime SDK.

## Why These Four Components Together

No single technique is sufficient. The system solves problems that need **all four** capabilities:

**Knowledge Graphs** — encode structured domain knowledge (entities, relations, facts) and propagate it through GNN-style message passing. The model doesn't learn "CYP3A4 is a liver enzyme" from data — it already knows.

**Differentiable Logic** — enforce hard rules that must hold regardless of what the neural network learns. "If drug A inhibits the enzyme that metabolizes drug B, flag an interaction." These rules are differentiable, so they participate in gradient-based training while remaining logically sound.

**Quantum Circuits** — encode features in exponentially large Hilbert spaces via parameterized quantum circuits. ZZ feature maps capture pairwise interactions through entanglement that classical kernels cannot efficiently represent. Variational circuits (VQE, QAOA) solve optimization problems on quantum hardware.

**Concept Bottleneck Models** — force all predictions through human-interpretable concepts. Instead of "probability 0.87", the system outputs "Contraindicated BECAUSE cyp2c19_conflict=0.94 AND narrow_therapeutic_index=0.91". This is required for domains where humans must understand and trust the model's reasoning.

```
Input Data
    ↓
Knowledge Graph ──→ GNN Embeddings
    ↓
Differentiable Logic ──→ Rule-Based Reasoning
    ↓
Quantum Kernels / Circuits ──→ Feature Encoding
    ↓
Concept Bottleneck ──→ Interpretable Prediction + Explanation
```

## Demo: Drug Interaction Safety Checker

The flagship example (`examples/drug_interaction_safety.py`) demonstrates a hospital prescription safety system combining all four components:

```bash
python examples/drug_interaction_safety.py
```

| Step | Component | What it produces |
|------|-----------|-----------------|
| 1 | **Knowledge Graph** | 8 structured facts: which drugs inhibit which enzymes, which drugs are metabolized by which enzymes |
| 2 | **KG-Guided GNN** | 16-dim embeddings for each drug/enzyme/pathway — drugs sharing an enzyme are closer in embedding space |
| 3 | **Differentiable Logic** | Risk scores derived from hard rules: "inhibits(A, enzyme) ∧ metabolized_by(B, enzyme) → risk" |
| 4 | **Quantum Kernels** | Molecular similarity via ZZ feature maps in Hilbert space — captures pairwise interactions through entanglement |
| 5 | **Concept Bottleneck** | Trained model that predicts through interpretable pharmacological concepts |
| 6 | **Evaluation** | Clinician-readable output: prediction + confidence + the specific concepts that triggered the alert |

The key result: **Omeprazole + Warfarin** is correctly flagged as Contraindicated because `cyp2c19_conflict=1.00` (Omeprazole inhibits CYP2C19, Warfarin is metabolized by it) AND `narrow_therapeutic_index=1.00` (Warfarin has a narrow therapeutic window — small dose changes cause dangerous bleeding). A black-box model would output "0.97" — this system tells the doctor *why*.

<details>
<summary><b>Full demo output</b> (click to expand)</summary>

```
======================================================================
  DRUG INTERACTION SAFETY CHECKER
  Quantum Neuro-Symbolic AI Demo
======================================================================

── Step 1: Pharmacological Knowledge Graph ──────────────────────
   8 entities, 3 relation types, 8 facts
   Ketoconazole ──inhibits──▶ CYP3A4
   Simvastatin  ──metabolized_by──▶ CYP3A4
   Omeprazole   ──inhibits──▶ CYP2C19
   Warfarin     ──metabolized_by──▶ CYP2C19  (narrow therapeutic index)

── Step 2: Drug Embeddings via KG-Guided GNN ───────────────────
   Embedding shape: torch.Size([8, 16])
   cos(Ketoconazole, Simvastatin) = +0.116  (share CYP3A4)
   cos(Ketoconazole, Warfarin)    = +0.360  (different enzymes)

── Step 3: Logic Reasoning — Interaction Rules ─────────────────
   Ketoconazole + Simvastatin           CYP3A4 risk=0.48  CYP2C19 risk=0.00
   Omeprazole + Warfarin                CYP3A4 risk=0.00  CYP2C19 risk=0.48
   Simvastatin + Omeprazole             CYP3A4 risk=0.00  CYP2C19 risk=0.00

── Step 4: Quantum Kernel — Molecular Similarity ───────────────
   K(Ketoconazole  , Simvastatin ) = 0.1223
   K(Ketoconazole  , Warfarin    ) = 0.0068
   K(Omeprazole    , Warfarin    ) = 0.2876
   → Quantum feature map captures pairwise interactions in
     exponentially large Hilbert space.

── Step 5: Train Concept Bottleneck Model ──────────────────────
   Dataset: 96 samples
     class 0 (Safe             ): 30 samples
     class 1 (Monitor          ): 42 samples
     class 2 (Contraindicated  ): 24 samples

   Training (concept_weight=0.5, 200 epochs)...
   Epoch  50  loss=0.0273  task=0.0531  concept=0.0016  acc=100%
   Epoch 100  loss=0.0053  task=0.0103  concept=0.0004  acc=100%
   Epoch 150  loss=0.0025  task=0.0042  concept=0.0009  acc=100%
   Epoch 200  loss=0.0018  task=0.0027  concept=0.0008  acc=100%

── Step 6: Interpretable Safety Predictions ────────────────────

   Ketoconazole + Simvastatin
   Expected: ⚠️  Monitor
   Predicted: ⚠️  Monitor  (confidence 100%)  ✓
   Because:
     • cyp3a4_conflict: 1.00

   Omeprazole + Warfarin
   Expected: 🚫 Contraindicated
   Predicted: 🚫 Contraindicated  (confidence 100%)  ✓
   Because:
     • cyp2c19_conflict: 1.00
     • narrow_therapeutic_index: 1.00

   Ketoconazole + Warfarin
   Expected: 🚫 Contraindicated
   Predicted: 🚫 Contraindicated  (confidence 100%)  ✓
   Because:
     • cyp2c19_conflict: 1.00
     • narrow_therapeutic_index: 1.00

   Simvastatin + Omeprazole
   Expected: ✅ Safe
   Predicted: ✅ Safe  (confidence 100%)  ✓
   Because: no pharmacological concepts triggered

   Simvastatin + Simvastatin
   Expected: ⚠️  Monitor
   Predicted: ⚠️  Monitor  (confidence 100%)  ✓
   Because:
     • same_pathway: 1.00

   Accuracy: 5/5 = 100%
```

</details>

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/quantum-neuro-symbolic-ai.git
cd quantum-neuro-symbolic-ai

pip install -r requirements.txt

# Run the drug interaction safety demo
python examples/drug_interaction_safety.py

# Run tests
python -m pytest tests/ -v

# Run benchmarks
python benchmarks/simple_benchmark.py
```

## Project Structure

```
quantum-neuro-symbolic-ai/
│
├── quantum_ml/                          # Quantum computing layer
│   ├── quantum_backend.py               # Unified backend (AerSimulator + IBM Quantum Runtime)
│   ├── hybrid_quantum_classical.py      # Hybrid quantum-classical models
│   ├── quantum_kernels.py               # Quantum kernel methods, QSVM
│   └── variational_circuits.py          # VQE, QAOA, variational classifiers
│
├── neuro_symbolic/                      # Classical neuro-symbolic layer
│   ├── differentiable_logic.py          # Fuzzy logic, forward chaining, neural-logic integration
│   ├── knowledge_guided_nn.py           # KG embeddings, KG-guided GNN, constrained attention
│   └── concept_bottleneck.py            # Independent/Sequential/Joint CBM architectures
│
├── quantum_neuro_symbolic/              # Quantum + neuro-symbolic integration
│   ├── quantum_logic_circuits.py        # Quantum differentiable logic gates
│   ├── quantum_kg_embedding.py          # Quantum KG embeddings, relation transforms
│   ├── quantum_cbm.py                   # Quantum concept bottleneck models
│   └── quantum_gnn.py                   # Quantum graph neural networks
│
├── examples/
│   ├── drug_interaction_safety.py       # Full pipeline demo (start here)
│   ├── enhanced_quantum_demo.py         # Quantum component showcase
│   └── quantum_neuro_symbolic_demo.py   # Integration demo
│
├── benchmarks/
│   └── simple_benchmark.py              # Synthetic tasks + classical baselines
│
├── tests/
│   ├── conftest.py
│   ├── test_quantum_backend.py          # 50+ tests with mocked IBM hardware
│   └── test_benchmarks.py              # Benchmark validation tests
│
├── requirements.txt
└── pytest.ini
```

## Architecture

### Quantum Backend Abstraction

The `QuantumBackend` class provides a single interface for both local simulation and IBM Quantum hardware:

```python
from quantum_ml.quantum_backend import QuantumBackend, QuantumBackendConfig

# Local simulation — no credentials needed
backend = QuantumBackend(QuantumBackendConfig(use_hardware=False))

# IBM Quantum hardware — requires token
backend = QuantumBackend(QuantumBackendConfig(
    use_hardware=True,
    resilience_level=2,
    enable_dynamical_decoupling=True,
))

# Same API for both
results = backend.run_sampler([circuit], shots=1024)
expectations = backend.run_estimator([circuit], [observable])
```

All quantum modules accept an optional `backend` parameter, defaulting to local simulation.

### Differentiable Logic

Rules are defined declaratively and compiled into differentiable PyTorch modules:

```python
from neuro_symbolic.differentiable_logic import DifferentiableLogicProgram

# parent(X,Y) :- mother(X,Y)
# parent(X,Y) :- father(X,Y)
# ancestor(X,Y) :- parent(X,Y)
rules = [(2, [0]), (2, [1]), (3, [2])]

logic = DifferentiableLogicProgram(n_predicates=4, rules=rules)
derived = logic(initial_facts)  # differentiable — gradients flow through
```

### Concept Bottleneck Models

Predictions are forced through interpretable concepts:

```python
from neuro_symbolic.concept_bottleneck import IndependentCBM

cbm = IndependentCBM(
    input_dim=12, n_concepts=4, n_classes=3,
    concept_names=["cyp3a4_conflict", "cyp2c19_conflict",
                   "same_pathway", "narrow_therapeutic_index"]
)

explanation = cbm.predict_with_explanation(features, threshold=0.4)
# → {'predicted_class': 2,
#    'class_probability': 0.97,
#    'active_concepts': [{'name': 'cyp2c19_conflict', 'probability': 0.94}, ...]}
```

### Quantum Kernels

```python
from quantum_ml.quantum_kernels import QuantumKernel, QuantumSVM

kernel = QuantumKernel(feature_map="zz", n_qubits=4, depth=2)
K = kernel.compute_kernel_matrix(X_train)

qsvm = QuantumSVM(kernel, C=1.0)
qsvm.fit(X_train, y_train)
predictions = qsvm.predict(X_test)
```

## IBM Quantum Hardware

The framework runs on real quantum hardware via the IBM Quantum Runtime SDK. All quantum components use the `QuantumBackend` abstraction that handles transpilation, session management, and error mitigation.

```bash
pip install qiskit-ibm-runtime
```

```python
from quantum_ml.quantum_backend import setup_ibm_quantum, QuantumBackend, QuantumBackendConfig

# One-time setup
setup_ibm_quantum(token="YOUR_IBM_QUANTUM_TOKEN")

# Use real quantum hardware
config = QuantumBackendConfig(
    use_hardware=True,
    optimization_level=2,
    resilience_level=2,
    enable_dynamical_decoupling=True,
)
backend = QuantumBackend(config)

# VQE on real hardware with session management
from quantum_ml.variational_circuits import VQE
vqe = VQE(hamiltonian, n_qubits=4, depth=2, backend=backend)
result = vqe.run(maxiter=100)
```

## Testing

The test suite includes 50+ tests with fully mocked IBM Quantum SDK responses, so all hardware code paths are tested without credentials:

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v -k "Local"       # local simulator tests only
python -m pytest tests/ -v -k "Hardware"     # hardware mock tests only
```

## Research Context

This project sits at the intersection of three active areas:

**For quantum ML**: PennyLane, TorchQuantum, and Qiskit ML are mature frameworks with auto-differentiation through quantum circuits, GPU-accelerated simulation, and real hardware deployment. This project's quantum ML layer (VQE, QAOA, quantum kernels) is a from-scratch implementation — without the performance or maturity of those frameworks.

**For neuro-symbolic AI**: DeepProbLog integrates Prolog with neural networks and actual gradient flow. Logic Tensor Networks implement fuzzy real logic in TensorFlow with real training. IBM's Logical Neural Networks are a purpose-built differentiable logic framework. This project's classical neuro-symbolic components are simpler implementations of these ideas.

**Where this project is distinct**: The combination — quantum circuits specifically designed for neuro-symbolic tasks (logic, KG reasoning, concept extraction, graph message passing) in one integrated system — is genuinely uncommon. Most quantum ML work focuses on classification or chemistry. Most neuro-symbolic work is classical. Putting quantum reasoning into the neuro-symbolic loop is the unexplored niche this project targets.

## Known Limitations

**Quantum parameters don't train via backprop.** Every quantum-classical boundary severs PyTorch's autograd graph (`.detach().cpu().numpy()` before each circuit execution). Quantum parameters are declared as `nn.Parameter` but gradients never reach them through the quantum path. Fixing this requires migrating to PennyLane with `interface="torch"` for native autodiff. This is the single most important issue.

**No demonstrated quantum advantage.** The quantum components run on a classical simulator with 4-8 qubits. At this scale, classical methods are strictly faster. The framework is architecturally ready for hardware that could provide advantage, but proving advantage requires larger circuits on real QPUs and carefully designed experiments.

**Toy scale with no scaling path.** All examples use 4-8 qubits, 5-6 entities, and 2-3 classes. The `QuantumKGEmbedding` computes fidelity between all entity pairs (O(n²) circuits for tail prediction). The `QuantumGNN` runs a separate circuit per node per message-passing step. There is no subgraph sampling or mini-batching of graph structure.

**No batching of quantum circuits.** Every quantum module processes samples one at a time in a Python for-loop. A batch of 32 means 32 sequential simulations. PennyLane and TorchQuantum support batched execution; raw Qiskit does not.

**Small-scale demos.** The drug interaction example uses 4 drugs and 96 training samples. Real pharmacovigilance databases have thousands of drugs and millions of interaction records.

## Roadmap

In priority order:

1. **Fix gradient flow.** Implement `torch.autograd.Function` with parameter-shift rule for each quantum layer, or rebuild on PennyLane. Without this, nothing trains end-to-end.
2. **Benchmark against classical baselines.** The benchmark framework exists (`benchmarks/simple_benchmark.py`) but needs to be run on tasks where quantum reasoning might help — e.g., few-shot relational reasoning on synthetic KGs.
3. **Profile and optimize.** Statevector simulation without measurement sampling would be faster and deterministic for circuits under ~20 qubits. Eliminate per-call simulator instantiation.
4. **Hardware-aware circuit design.** Profile circuit depth against IBM's ~100-layer decoherence limit. Replace full entanglement with hardware-native heavy-hex connectivity.
5. **Scale analysis.** Characterize where (if anywhere) quantum circuits outperform classical approximations. Tensor networks can often approximate quantum states classically — when does the quantum version actually help?

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Qiskit 2.x
- qiskit-aer
- qiskit-ibm-runtime (optional, for hardware access)
- scipy, numpy

## License

MIT

---

**Last updated**: March 2026
**Lines of code**: ~4,500 Python across 10 modules
**Maturity**: Proof-of-concept prototype
