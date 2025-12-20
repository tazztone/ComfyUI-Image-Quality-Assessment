#!/usr/bin/env python
"""
ComfyUI-Image-Quality-Assessment Test Runner

This script runs pytest with proper environment isolation to prevent
the main __init__.py from being loaded during unit tests.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py -m unit            # Run only unit tests  
    python run_tests.py -m integration     # Run only integration tests
"""

import os
import sys
import subprocess

# Set testing environment BEFORE any imports
os.environ['COMFYUI_TESTING'] = '1'

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, '..', '..', 'venv', 'Scripts', 'python.exe')

# Build pytest command
pytest_args = [
    venv_python,
    '-m', 'pytest',
    '.',  # Current directory (tests/)
    '-v',
    '--tb=short',
]

# Add any additional arguments passed to this script
pytest_args.extend(sys.argv[1:])

# Run from tests directory to avoid package import
tests_dir = os.path.join(script_dir, 'tests')
os.chdir(tests_dir)

print(f"Running: {' '.join(pytest_args)}")
result = subprocess.run(pytest_args, env=os.environ)
sys.exit(result.returncode)
