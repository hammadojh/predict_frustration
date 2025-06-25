#!/bin/bash

# GitHub Setup Script for One-Turn-Ahead Frustration Prediction
# This script prepares the project for GitHub by initializing git and setting up the repository

echo "🚀 Setting up One-Turn-Ahead project for GitHub..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files to staging (respecting .gitignore)
echo "📋 Adding files to Git (respecting .gitignore)..."
git add .

# Show status
echo "📊 Git status:"
git status --short

# Show what will be ignored
echo ""
echo "🚫 Files that will be ignored (sample):"
echo "   - venv/ (virtual environment)"
echo "   - data/*.jsonl (large dataset files)"
echo "   - checkpoints/ (model files)"
echo "   - results/*.json (experiment results)"
echo "   - __pycache__/ (Python cache)"

echo ""
echo "📝 Ready for initial commit! Run:"
echo "   git commit -m 'Initial commit: M1 BERT-CLS baseline with exceptional results'"
echo ""
echo "🌐 To push to GitHub:"
echo "   1. Create a new repository on GitHub"
echo "   2. git remote add origin https://github.com/yourusername/one_turn_ahead.git"
echo "   3. git branch -M main"
echo "   4. git push -u origin main"
echo ""
echo "✅ Project is ready for GitHub!" 