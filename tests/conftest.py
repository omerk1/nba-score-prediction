"""
Pytest configuration file for NBA Score Prediction tests.

Sets up the Python path so that src module can be imported.
"""

import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
