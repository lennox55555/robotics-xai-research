#!/bin/bash
# Setup script for the research project

set -e

echo "Setting up robotics-xai-research environment..."

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify W&B setup
echo "Verifying W&B configuration..."
source .env
if [ -n "$WANDB_API_KEY" ]; then
    echo "W&B API key found"
    wandb login --relogin $WANDB_API_KEY
else
    echo "Warning: WANDB_API_KEY not found in .env"
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run an experiment:"
echo "  python experiments/mujoco/train.py"
