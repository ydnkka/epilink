# Quick Reference: TestPyPI Versioning

## Problem & Solution

**Problem:** TestPyPI rejects duplicate versions (400 Bad Request)
**Solution:** Auto-generate unique dev versions for each test release

---

## Version Numbering

| Method | Version Format | Example |
|--------|----------------|---------|
| GitHub Actions | `0.0.0.dev{run_number}` | `0.0.0.dev123` |
| Manual (script) | `0.0.0.dev{timestamp}` | `0.0.0.dev1709395200` |
| Manual (DIY) | `0.0.0.dev{timestamp}` | `0.0.0.dev1709395200` |
| Production | `{major}.{minor}.{patch}` | `1.0.0` |

---

## Quick Commands

### Test Release via GitHub Actions
```bash
# Just click "Run workflow" in GitHub Actions → Test Release
# Version automatically increments: 0.0.0.dev1, 0.0.0.dev2, 0.0.0.dev3...
```

### Test Release Manually (Recommended)
```bash
./scripts/test_release.sh
# Then upload: twine upload --repository testpypi dist/*
```

### Test Release Manually (Full Control)
```bash
# Create unique version tag
git tag -a "v0.0.0.dev$(date +%s)" -m "Test release"

# Build
python -m build

# Verify
twine check dist/*

# Clean up tag
git tag -d v0.0.0.dev*

# Upload
twine upload --repository testpypi dist/*
```

### Install from TestPyPI
```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    epilink
```

### Check Available Versions
```bash
pip index versions epilink --index-url https://test.pypi.org/simple/
```

---

## Workflow Logic

```
GitHub Actions Trigger
       ↓
Create tag: v0.0.0.dev{run_number}
       ↓
hatch-vcs detects tag
       ↓
Build version: 0.0.0.dev{run_number}
       ↓
Upload to TestPyPI ✅
       ↓
Each new run → new number → no conflicts!
```

---

## Clean Up

### Remove local build artifacts
```bash
rm -rf dist/ build/ *.egg-info
```

### Remove test tags (if needed)
```bash
git tag -d v0.0.0.dev*
```

---

## Production Release (When Ready)

```bash
# Tag with real version
git tag v1.0.0

# Push tag
git push origin v1.0.0

# GitHub Actions automatically publishes to PyPI via release.yml
```

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| 400 Bad Request | Duplicate version | ✅ Fixed! Auto-incrementing now |
| File already exists | Same version uploaded | Use script or GitHub Actions |
| Version is 0.0.0 | No git tags | ✅ Fixed! Creates tags automatically |
| Can't find package | Wrong index URL | Use both --index-url and --extra-index-url |

---

## Files Updated

- ✅ `.github/workflows/test-release.yml` - Auto versioning
- ✅ `scripts/test_release.sh` - Timestamp-based versioning
- ✅ `TESTPYPI.md` - Full documentation
- ✅ `RELEASE_CHECKLIST.md` - Updated checklist

**Everything is ready! Try it now.** 🚀

