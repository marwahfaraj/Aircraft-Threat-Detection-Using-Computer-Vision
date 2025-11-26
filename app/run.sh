#!/bin/bash
# Quick start script for Aircraft Threat Detection App

echo "Starting Aircraft Threat Detection App..."
echo "=========================================="

# Check if we're in the app directory
if [ ! -f "app.py" ]; then
    echo "Error: Please run this script from the app/ directory"
    echo "Or run: python app/app.py from project root"
    exit 1
fi

# Run the app
python app.py

