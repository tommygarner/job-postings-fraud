#!/bin/bash

echo "Setting up Fake Job Posting Detection Project..."
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install requirements
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Download dataset:"
echo "   kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction"
echo "   unzip real-or-fake-fake-jobposting-prediction.zip -d data/raw/"
echo "3. Start coding!"
