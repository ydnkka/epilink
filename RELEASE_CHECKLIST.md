# TestPyPI Release Checklist

## First Time Setup

- [ ] Create account at https://test.pypi.org/account/register/
- [ ] Configure Trusted Publishing at https://test.pypi.org/manage/account/publishing/
  - PyPI Project Name: `epilink`
  - Owner: `ydnkka`
  - Repository: `epilink`
  - Workflow: `test-release.yml`
  - Environment: `testpypi`
- [ ] Create GitHub Environment named `testpypi` in repo settings

## Before Each Test Release

- [ ] All tests passing locally: `pytest`
- [ ] Code linted: `ruff check .` and `black --check .`
- [ ] Type checking clean: `mypy src/epilink`
- [ ] CHANGELOG updated (if applicable)
- [ ] ✅ Version will be auto-generated (no manual action needed!)

**Note:** Versions are automatically unique:
- GitHub Actions: `0.0.0.dev{run_number}` (auto-incrementing)
- Manual script: `0.0.0.dev{timestamp}` (uses current time)

## Release via GitHub Actions (Recommended)

1. [ ] Go to GitHub Actions tab
2. [ ] Select "Test Release" workflow
3. [ ] Click "Run workflow"
4. [ ] (Optional) Enter version suffix
5. [ ] Click green "Run workflow" button
6. [ ] Wait for workflow to complete
7. [ ] Check https://test.pypi.org/project/epilink/

## Release Manually (Alternative)

```bash
# Clean previous builds
rm -rf dist/

# Build
python -m build

# Check
twine check dist/*

# Upload
twine upload --repository testpypi dist/*
```

## Test Installation

```bash
# Create test environment
python -m venv test-env
source test-env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    epilink

# Test basic functionality
epilink --help
python -c "import epilink; print(epilink.__version__)"

# Run a quick test
# ... your test commands here ...

# Cleanup
deactivate
rm -rf test-env
```

## Verification Checklist

- [ ] Package installs without errors
- [ ] Version number is correct
- [ ] CLI command works: `epilink --help`
- [ ] Python import works: `import epilink`
- [ ] Dependencies installed correctly
- [ ] Basic functionality works

## Ready for Production Release?

If all tests pass on TestPyPI:

```bash
# Tag the release
git tag v1.0.0  # Use actual version number

# Push the tag
git push origin v1.0.0

# GitHub Actions will automatically publish to PyPI
```

---

## Quick Commands Reference

```bash
# Manual test release (all steps)
./scripts/test_release.sh

# Just build
python -m build

# Just check
twine check dist/*

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ epilink

# Clean build artifacts
rm -rf dist/ build/ *.egg-info
```

