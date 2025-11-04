#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
# Use Python 3.12 if available, otherwise fall back to python3
if command -v python3.12 &> /dev/null; then
    python3.12 -m venv venv
else
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Setup complete! To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then run the web server:"
echo "  python app.py"

