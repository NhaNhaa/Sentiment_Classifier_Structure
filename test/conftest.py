"""Pytest configuration - adds parent directory to Python path."""

import sys
import os

# Add the parent directory (project root) to Python path
# This allows 'import config' to work in test files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))