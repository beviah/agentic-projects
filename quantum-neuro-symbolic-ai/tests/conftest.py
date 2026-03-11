"""Pytest configuration — adds project root to sys.path."""

import sys
import os

# Add project root so `from quantum_ml.…` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
