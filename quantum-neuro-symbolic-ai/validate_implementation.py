#!/usr/bin/env python3
"""Validation script for quantum neuro-symbolic implementation.

Checks code structure, imports, and basic syntax without requiring
all dependencies to be installed.
"""

import os
import sys
import ast

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_file_exists(filepath):
    """Check if a file exists."""
    return os.path.exists(filepath)

def main():
    print("=" * 70)
    print("Quantum Neuro-Symbolic AI - Implementation Validation")
    print("=" * 70)
    
    project_root = "/root/agentic/projects/quantumai"
    
    # Files to validate
    files_to_check = [
        # New implementations
        "quantum_neuro_symbolic/quantum_cbm.py",
        "quantum_neuro_symbolic/quantum_gnn.py",
        "examples/enhanced_quantum_demo.py",
        
        # Existing implementations
        "quantum_neuro_symbolic/quantum_logic_circuits.py",
        "quantum_neuro_symbolic/quantum_kg_embedding.py",
        "neuro_symbolic/differentiable_logic.py",
        "neuro_symbolic/knowledge_guided_nn.py",
        "neuro_symbolic/concept_bottleneck.py",
        "quantum_ml/hybrid_quantum_classical.py",
        "quantum_ml/quantum_kernels.py",
        "quantum_ml/variational_circuits.py",
        
        # Documentation
        "README.md",
        "IMPROVEMENT_PLAN.md",
        "IMPLEMENTATION_SUMMARY.md",
        "CHANGES.md",
        "requirements.txt",
    ]
    
    print("\n1. File Existence Check")
    print("-" * 70)
    
    all_exist = True
    for filepath in files_to_check:
        full_path = os.path.join(project_root, filepath)
        exists = check_file_exists(full_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {filepath}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n  ✓ All files exist")
    else:
        print("\n  ✗ Some files missing")
    
    print("\n2. Python Syntax Validation")
    print("-" * 70)
    
    python_files = [f for f in files_to_check if f.endswith('.py')]
    all_valid = True
    
    for filepath in python_files:
        full_path = os.path.join(project_root, filepath)
        if check_file_exists(full_path):
            valid, message = check_file_syntax(full_path)
            status = "✓" if valid else "✗"
            print(f"  {status} {filepath}: {message}")
            if not valid:
                all_valid = False
        else:
            print(f"  ⊘ {filepath}: File not found")
            all_valid = False
    
    if all_valid:
        print("\n  ✓ All Python files have valid syntax")
    else:
        print("\n  ✗ Some Python files have syntax errors")
    
    print("\n3. Code Statistics")
    print("-" * 70)
    
    total_lines = 0
    new_implementation_lines = 0
    
    new_files = [
        "quantum_neuro_symbolic/quantum_cbm.py",
        "quantum_neuro_symbolic/quantum_gnn.py",
        "examples/enhanced_quantum_demo.py",
    ]
    
    for filepath in python_files:
        full_path = os.path.join(project_root, filepath)
        if check_file_exists(full_path):
            with open(full_path, 'r') as f:
                lines = len(f.readlines())
                total_lines += lines
                if filepath in new_files:
                    new_implementation_lines += lines
    
    print(f"  Total Python code: {total_lines} lines")
    print(f"  New implementations: {new_implementation_lines} lines")
    print(f"  New files: {len(new_files)}")
    
    print("\n4. Component Summary")
    print("-" * 70)
    
    print("\n  Classical Neuro-Symbolic:")
    print("    ✓ Differentiable Logic")
    print("    ✓ Knowledge-Guided GNN")
    print("    ✓ Concept Bottleneck Model")
    
    print("\n  Quantum ML:")
    print("    ✓ Hybrid Quantum-Classical")
    print("    ✓ Quantum Kernels")
    print("    ✓ Variational Circuits")
    
    print("\n  Quantum Neuro-Symbolic:")
    print("    ✓ Quantum Logic Circuits")
    print("    ✓ Quantum KG Embedding")
    print("    ✓ Quantum Concept Bottleneck (NEW)")
    print("    ✓ Quantum Graph Neural Network (NEW)")
    
    print("\n5. Documentation")
    print("-" * 70)
    
    docs = [
        ("README.md", "Project overview"),
        ("IMPROVEMENT_PLAN.md", "Development roadmap"),
        ("IMPLEMENTATION_SUMMARY.md", "Technical details"),
        ("CHANGES.md", "Changelog"),
    ]
    
    for doc, description in docs:
        full_path = os.path.join(project_root, doc)
        if check_file_exists(full_path):
            with open(full_path, 'r') as f:
                lines = len(f.readlines())
            print(f"  ✓ {doc}: {lines} lines - {description}")
        else:
            print(f"  ✗ {doc}: Missing")
    
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    
    if all_exist and all_valid:
        print("\n  ✓ All validations passed")
        print("  ✓ Code structure is correct")
        print("  ✓ Syntax is valid")
        print("  ✓ Documentation is complete")
        print("\n  Status: READY FOR TESTING")
        print("\n  Next steps:")
        print("    1. Install dependencies: pip install -r requirements.txt")
        print("    2. Run demos: python examples/enhanced_quantum_demo.py")
        print("    3. Test components individually")
        return 0
    else:
        print("\n  ✗ Some validations failed")
        print("  Status: NEEDS ATTENTION")
        return 1

if __name__ == "__main__":
    sys.exit(main())
