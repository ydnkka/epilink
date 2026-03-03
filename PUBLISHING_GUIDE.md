# Publishing Guide: PyPI & TestPyPI

A comprehensive beginner-friendly guide to publishing Python packages on PyPI and TestPyPI.

---

## Table of Contents

1. [Overview](#overview)
2. [Pre-Publishing Setup](#pre-publishing-setup)
3. [Testing with TestPyPI](#testing-with-testpypi)
4. [Publishing to PyPI](#publishing-to-pypi)
5. [Troubleshooting](#troubleshooting)

---

## Overview

### What is PyPI?

**PyPI** (Python Package Index) is the official repository where Python packages are stored. When you run `pip install package_name`, Python downloads from PyPI.

**TestPyPI** is a practice environment that works exactly like PyPI but is isolated for testing. It's safe to experiment here without affecting real packages.

### Publication Workflow

```
Your Code
    ↓
Build Package
    ↓
Test on TestPyPI (optional but recommended)
    ↓
Publish to PyPI
    ↓
Users can: pip install your-package
```

Recommended approach: use GitHub Actions with Trusted Publisher (OIDC) for CI. Token-based uploads are optional for manual runs only.

---

## Pre-Publishing Setup

### 1. Prepare Your Project Structure

Ensure your project follows this structure:

```
my-project/
├── pyproject.toml          # Main configuration (✅ most important)
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── ...
├── tests/
│   └── ...
├── README.md               # Project description
├── LICENSE                 # License file (MIT, Apache-2.0, etc.)
└── .gitignore
```

### 2. Configure `pyproject.toml`

The `pyproject.toml` file is crucial. It defines:
- Package name
- Version
- Dependencies
- Metadata (description, author, etc.)

**Minimal Example:**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-package"                    # Must be unique on PyPI
version = "0.1.0"                      # Semantic versioning: major.minor.patch
description = "A short description"
readme = "README.md"
requires-python = ">=3.8"
authors = [
    { name = "Your Name", email = "you@example.com" }
]
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
]

[project.urls]
Homepage = "https://github.com/yourusername/my-package"
Documentation = "https://mypackage.readthedocs.io"
```

**Version Strategies:**

- **Static versioning (recommended):** Set explicitly in `pyproject.toml`: `version = "1.0.0"`
  ```toml
  [project]
  name = "my-package"
  version = "1.0.0"
  ```
  
- **Dynamic versioning (optional):** Use `hatch-vcs` to auto-detect from git tags
  ```toml
  [build-system]
  requires = ["hatchling>=1.22", "hatch-vcs>=0.4"]
  
  [project]
  name = "my-package"
  dynamic = ["version"]

  [tool.hatch.version]
  source = "vcs"

  [tool.hatch.build.hooks.vcs]
  version-file = "src/my_package/_version.py"
  ```
  Tags can be `v1.2.3` or `1.2.3`.

**Note:** This project (epilink) uses **static versioning**. Update the version in `pyproject.toml` before each release.

### 3. Create Required Files

**README.md**
- First 100 words should hook the reader
- Include installation instructions
- Show minimal usage example

**LICENSE**
- Required for distribution
- Common choices: MIT, Apache-2.0, GPL-3.0
- Add file to root: `LICENSE` or `LICENSE.md`

**Example README:**

```markdown
# My Package

Fast and easy way to do amazing things!

## Installation

```bash
pip install my-package
```

## Quick Start

```python
from my_package import do_something

result = do_something(data)
print(result)
```

### 4. Create PyPI & TestPyPI Accounts

1. **TestPyPI:** https://test.pypi.org/account/register/
2. **PyPI:** https://pypi.org/account/register/

Use the same email/username for consistency.

### 5. Configure Trusted Publisher (OIDC)

Trusted Publisher uses GitHub OIDC to mint short-lived upload tokens. No long-lived API tokens are stored in GitHub secrets.

1. In TestPyPI and PyPI:
   - Open your project page -> **Manage** -> **Publishing**.
   - Add a **Trusted Publisher** for GitHub Actions.
   - Set **Owner**, **Repository**, **Workflow filename**, and **Environment**.
     - Example workflow files: `test-release.yml` (TestPyPI) and `release.yml` (PyPI)
     - Example environments: `testpypi` and `pypi`

2. In GitHub:
   - Ensure your workflows use matching `environment` names.
   - Ensure the workflow has `permissions: id-token: write`.

Note: Trusted Publisher entries are separate for TestPyPI and PyPI.

---

## Testing with TestPyPI

Always test on TestPyPI before publishing to PyPI. You can't delete versions from PyPI!

### Method 1: Using GitHub Actions (Trusted Publisher, Recommended)

1. Ensure Trusted Publisher is configured in TestPyPI for this repo, workflow file, and environment.

2. Create `.github/workflows/test-release.yml`:

```yaml
name: Test Release

on:
  workflow_dispatch:
    inputs:
      version_suffix:
        description: 'PEP 440 suffix (e.g., dev123, rc1, b1). Leave blank for dev<run_number>.'
        required: false
        default: ''

permissions:
  contents: read
  id-token: write

jobs:
  publish-testpypi:
    runs-on: ubuntu-latest
    environment:
      name: testpypi
      url: https://test.pypi.org/p/your-project
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Set test version
        run: |
          python -m pip install --upgrade pip tomlkit
          SUFFIX="${{ inputs.version_suffix }}"
          if [ -z "$SUFFIX" ]; then
            SUFFIX="dev${GITHUB_RUN_NUMBER}"
          fi

          export EPILINK_SUFFIX="$SUFFIX"
          python - <<'PY'
          import os
          from pathlib import Path
          import tomlkit

          path = Path("pyproject.toml")
          text = path.read_text(encoding="utf-8")
          data = tomlkit.parse(text)
          project = data.get("project")
          if not project or "version" not in project:
            raise SystemExit("version not found in pyproject.toml")

          base_version = str(project["version"])
          suffix = os.environ["EPILINK_SUFFIX"]
          new_version = f"{base_version}.{suffix}"

          project["version"] = new_version
          path.write_text(tomlkit.dumps(data), encoding="utf-8")
          print(f"Using test version: {new_version}")
          PY

      - name: Build distributions
        run: |
          python -m pip install --upgrade pip build twine
          python -m build
          python -m twine check dist/*

      - name: Publish to TestPyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
```

3. Run Workflow:
   - GitHub -> **Actions** -> **Test Release**
   - Click **Run workflow**

### Method 2: Manual Upload (Token-based, Optional)

This is useful for one-off local uploads. For CI, prefer Trusted Publisher.

Create a TestPyPI API token in your account settings before starting.

**Step 1: Install tools**

```bash
pip install build twine
```

**Step 2: Build the package**

```bash
python -m build
```

Creates:
- `dist/my-package-0.1.0.tar.gz` (source distribution)
- `dist/my_package-0.1.0-py3-none-any.whl` (wheel/binary)

**Step 3: Check the package**

```bash
twine check dist/*
```

Validates metadata before upload.

**Step 4: Upload to TestPyPI**

```bash
twine upload --repository testpypi dist/*
```

When prompted, enter:
- Username: `__token__`
- Password: (paste your TestPyPI token)

**Step 5: Test installation**

```bash
# In a fresh virtual environment
python -m venv test_env
source test_env/bin/activate

pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    my-package
```

The `--extra-index-url` gets dependencies from regular PyPI.

**Step 6: Verify it works**

```python
python -c "import my_package; print(my_package.__version__)"
```

---

## Publishing to PyPI

Only publish when:
- ✅ All tests pass locally
- ✅ Code is linted and formatted
- ✅ Version bump committed to git
- ✅ Release notes documented
- ✅ Tested on TestPyPI

### Semantic Versioning Explained

```
version = MAJOR.MINOR.PATCH

1.0.0   → Initial release
1.1.0   → New features (backward compatible)
1.1.1   → Bug fixes (backward compatible)
2.0.0   → Breaking changes
```

**When to bump:**
- **MAJOR:** Breaking API changes (users must update code)
- **MINOR:** New features, backward compatible
- **PATCH:** Bug fixes only
- **PRE-RELEASE:** Use suffixes like `1.0.0-beta1`, `1.0.0rc1`

### Publishing with GitHub Actions

**Important for epilink (static versioning):** When bumping the version, you must update it in **two places**:

1. **Update version in `pyproject.toml`:**

```toml
[project]
name = "epilink"
version = "1.0.0"  # Update this to your new version
```

2. **Update version in `src/epilink/__init__.py`:**

```python
__version__ = "1.0.0"  # Must match pyproject.toml
```

Commit both changes to your repository.

3. **Ensure Trusted Publisher is configured in PyPI for this repo, workflow file, and environment.**

4. **Create `.github/workflows/release.yml`:**

```yaml
name: Release to PyPI

on:
  release:
    types: [published]

permissions:
  contents: read
  id-token: write

jobs:
  publish:
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/your-project
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - run: python -m pip install --upgrade pip build twine
      
      - run: python -m build
      
      - name: Check distribution
        run: python -m twine check dist/*
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

5. **Trigger Release:**
   - Commit the version change to your repository
   - Push to GitHub
   - Go to **Releases** → **Create a new release**
   - Tag: `v1.0.0` (should match the version in pyproject.toml)
   - Title: `Release 1.0.0`
   - Click **Publish release**
   - Workflow runs automatically

### Publishing Manually (Token-based, Optional)

If you must upload locally, create a PyPI API token in your account settings and use `twine`. This is not recommended for CI.

```bash
# 1. Ensure clean working directory
git status

# 2. Build
python -m build

# 3. Check
twine check dist/*

# 4. Upload to PyPI
twine upload dist/*
```

When prompted, use `__token__` as username and your PyPI token as password.

### Verify Publication

1. **Check PyPI:** https://pypi.org/project/my-package/
2. **Test installation in clean environment:**

```bash
pip install my-package
```

3. **View release history:**
   - PyPI project page → **Release history**

---

## Troubleshooting

### Error: "Filename already exists"

**Cause:** Version already published (TestPyPI/PyPI doesn't allow re-uploads, even if the release was deleted)

**Solution:** Increment version and rebuild

```bash
# Edit pyproject.toml
version = "0.1.1"  # Changed from 0.1.0

python -m build
twine upload --repository testpypi dist/*
```

### Error: "Invalid metadata"

**Check with:**

```bash
twine check dist/*
```

**Common fixes:**
- Ensure `README.md` exists and is referenced in `pyproject.toml`
- Check email format in authors field
- Ensure package name is lowercase (with hyphens only)

### Error: "Package name unavailable"

**Cause:** Name already taken on PyPI

**Solution:** Change `name` in `pyproject.toml` to something unique:

```toml
name = "my-package-awesome-v2"
```

### Installation fails with "No module named 'my_package'"

**Causes:**
1. Package name vs. import name mismatch
2. Missing `__init__.py` in source folder

**Check:**

```bash
# Package name (in pyproject.toml)
name = "my-package"

# But import as (must have __init__.py)
import my_package  # or my_package.submodule
```

**Fix project structure:**

```
✅ Correct:
src/my_package/__init__.py
src/my_package/module.py

❌ Wrong:
my_package/__init__.py
my_package/module.py
```

### Slow upload

**Solution:** Use `.whl` format (wheels upload faster than source distributions)

```bash
# Uploaded automatically with python -m build
# Check that both .tar.gz and .whl exist in dist/
ls dist/
```

### Error: "Not authorized" / "Invalid or missing OIDC token"

**Cause:** Trusted Publisher not configured or workflow/environment mismatch. `id-token` permission may be missing.

**Solution:**
- Ensure the Trusted Publisher entry matches owner, repo, workflow file, and environment name.
- Ensure the workflow has `permissions: id-token: write`.

---

## Best Practices Checklist

- [ ] Package name is unique and lowercase
- [ ] `pyproject.toml` is well-configured
- [ ] `README.md` has usage examples
- [ ] `LICENSE` file present
- [ ] Version follows semantic versioning
- [ ] All tests pass: `pytest`
- [ ] Code is formatted: `black .`
- [ ] Code is linted: `ruff check .`
- [ ] Type checking passes: `mypy src/`
- [ ] Tested on TestPyPI first
- [ ] Git tags match release versions (e.g., `v1.0.0`)

---

## Quick Reference

| Task | Command |
|------|---------|
| Run TestPyPI publish (OIDC) | GitHub Actions -> Test Release |
| Run PyPI publish (OIDC) | Publish a GitHub Release |
| Build package | `python -m build` |
| Check metadata | `twine check dist/*` |
| Manual upload to TestPyPI (token) | `twine upload --repository testpypi dist/*` |
| Manual upload to PyPI (token) | `twine upload dist/*` |
| Check PyPI page | `https://pypi.org/project/YOUR_PACKAGE/` |
| View package files | `twine download package-name==1.0.0` |
| Configure Trusted Publisher (TestPyPI) | https://test.pypi.org/manage/project/YOUR_PACKAGE/ |
| Configure Trusted Publisher (PyPI) | https://pypi.org/manage/project/YOUR_PACKAGE/ |

---

## Additional Resources

- **Official PyPI Guide:** https://packaging.python.org/
- **Build Documentation:** https://build.pypa.io/
- **Twine Documentation:** https://twine.readthedocs.io/
- **Semantic Versioning:** https://semver.org/
- **PEP 440 (Versioning):** https://www.python.org/dev/peps/pep-0440/

---

## Questions?

If something isn't clear, check these in order:

1. This guide (use Ctrl+F to search)
2. Official PyPI documentation
3. Your project's `pyproject.toml` file
4. GitHub Issues for your project

Good luck publishing! 🚀
