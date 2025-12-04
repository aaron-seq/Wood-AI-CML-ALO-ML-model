#!/usr/bin/env python3
"""
Wood AI CML ALO ML Model - Complete Project Generator

This script generates the entire project structure with production-ready code.
Run this once after cloning the repository to set up all files.

Usage:
    python generate_project.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("\n" + "="*60)
print("Wood AI CML ALO ML Model - Project Generator")
print("="*60 + "\n")

# Create directory structure
dirs = [
    "api",
    "api/routes",
    "ml",
    "ml/models",
    "ml/preprocessing",
    "ml/training",
    "ml/utils",
    "data/raw",
    "data/processed",
    "data/training",
    "data/models",
    "notebooks",
    "tests",
    "scripts",
    "docs",
]

print("[1/4] Creating directory structure...")
for dir_path in dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {dir_path}/")

print("\n[2/4] Generating configuration files...")

# Dockerfile
with open("Dockerfile", "w") as f:
    f.write("""# Multi-stage build for Wood AI CML ALO ML Model
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
