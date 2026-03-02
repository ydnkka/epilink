#!/bin/bash

# Quick script to build and upload to TestPyPI
# Usage: ./scripts/test_release.sh

set -e  # Exit on error

echo "🔨 Building package..."
python -m pip install --upgrade build twine
python -m build

echo ""
echo "✅ Checking distribution..."
python -m twine check dist/*

echo ""
echo "📦 Contents of dist/:"
ls -lh dist/

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

