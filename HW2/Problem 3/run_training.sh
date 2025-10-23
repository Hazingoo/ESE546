#!/bin/bash

# Script to run All-CNN training with virtual environment
# Usage: ./run_training.sh

# Navigate to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
source ../../.venv/bin/activate

# Run the training script
python train_allcnn.py

# Deactivate virtual environment
deactivate
