#!/bin/bash

# Quick script to build and upload to TestPyPI
# Usage: ./scripts/test_release.sh

set -e  # Exit on error

echo "🔨 Preparing test release..."
echo ""

# Create a unique dev version tag for this build
TIMESTAMP=$(date +%s)
TEST_TAG="v0.0.0.dev${TIMESTAMP}"

echo "📝 Creating temporary version tag: $TEST_TAG"
git tag -a "$TEST_TAG" -m "Test release build at $(date)"

echo ""
echo "🔨 Building package..."
python -m pip install --upgrade build twine
python -m build

echo ""
echo "✅ Checking distribution..."
python -m twine check dist/*

echo ""
echo "📦 Contents of dist/:"
ls -lh dist/

# Clean up the temporary tag
echo ""
echo "🧹 Cleaning up temporary tag..."
git tag -d "$TEST_TAG"

echo ""
echo "🚀 Ready to upload to TestPyPI!"
echo ""
echo "Run one of the following commands:"
echo ""
echo "Option 1: Upload with interactive prompt for credentials"
echo "  twine upload --repository testpypi dist/*"
echo ""
echo "Option 2: Upload with token (set your token first)"
echo "  twine upload --repository testpypi dist/* --username __token__ --password \$TEST_PYPI_TOKEN"
echo ""
echo "After upload, test installation with:"
echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ epilink"
echo ""
echo "Note: Built version will be something like 0.0.0.dev${TIMESTAMP}"

