# Version Management Guide

This project uses **static versioning**. The version is manually set in the code files.

## How to Update the Version

When preparing a new release, update the version in **BOTH** of these files:

### 1. pyproject.toml

```toml
[project]
name = "epilink"
version = "0.1.0"  # ← Update this
```

### 2. src/epilink/__init__.py

```python
__version__ = "0.1.0"  # ← Update this (must match pyproject.toml)
```

## Version Numbering (Semantic Versioning)

Follow semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR** (e.g., 1.0.0 → 2.0.0): Breaking changes that require users to update their code
- **MINOR** (e.g., 1.0.0 → 1.1.0): New features, backward compatible
- **PATCH** (e.g., 1.0.0 → 1.0.1): Bug fixes, backward compatible

## Release Process

1. **Update both version files** (pyproject.toml and __init__.py)
2. **Commit the changes**: `git commit -am "Bump version to X.Y.Z"`
3. **Push to GitHub**: `git push`
4. **Create a GitHub Release**:
   - Tag: `vX.Y.Z` (e.g., `v1.0.0`)
   - Title: `Release X.Y.Z`
   - Description: List changes/fixes
5. **Publish the release** - This triggers the GitHub Actions workflow to publish to PyPI

## Testing Before Release

Before publishing to PyPI, test on TestPyPI:

1. Go to GitHub Actions
2. Select "Test Release" workflow
3. Click "Run workflow"
4. Optionally provide a version suffix (or leave blank for auto-generated dev version)

This will create a test version like `0.1.0.dev123` on TestPyPI.

## Quick Checklist

- [ ] Updated version in `pyproject.toml`
- [ ] Updated `__version__` in `src/epilink/__init__.py`
- [ ] Both versions match exactly
- [ ] Committed and pushed changes
- [ ] Tested on TestPyPI (optional but recommended)
- [ ] Created GitHub release with matching tag
- [ ] Verified publication on PyPI

