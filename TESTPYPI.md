# Testing Releases on TestPyPI

This guide shows you how to test package releases before publishing to the main PyPI repository.

## Option 1: Using GitHub Actions (Recommended for CI/CD)

### Setup (One-time)

1. **Create a TestPyPI account** (if you don't have one):
   - Go to https://test.pypi.org/account/register/
   - Verify your email

2. **Configure Trusted Publishing** (recommended, no API tokens needed):
   - Go to https://test.pypi.org/manage/account/publishing/
   - Click "Add a new pending publisher"
   - Fill in:
     - PyPI Project Name: `epilink`
     - Owner: `ydnkka` (your GitHub username)
     - Repository name: `epilink`
     - Workflow name: `test-release.yml`
     - Environment name: `testpypi`
   - Click "Add"

   **OR** use API tokens (alternative):
   - Go to https://test.pypi.org/manage/account/token/
   - Create a new API token with scope for "Entire account" or just "epilink"
   - Copy the token (starts with `pypi-`)
   - In your GitHub repo, go to Settings > Secrets and variables > Actions
   - Create a new repository secret named `TEST_PYPI_API_TOKEN`
   - Paste the token value

3. **Create GitHub Environment** (if using trusted publishing):
   - Go to your GitHub repo Settings > Environments
   - Click "New environment"
   - Name it `testpypi`
   - Click "Configure environment"
   - (Optional) Add protection rules if desired

### Trigger a Test Release

The workflow is triggered manually:

1. Go to your GitHub repository
2. Click on "Actions" tab
3. Select "Test Release" workflow from the left sidebar
4. Click "Run workflow" button (top right)
5. Optionally specify a version suffix (e.g., `rc1`, `beta1`, `test1`)
6. Click the green "Run workflow" button

The workflow will:
- Build your package
- Run twine check to validate
- Publish to TestPyPI

### Verify the Test Release

After the workflow completes:

1. Visit https://test.pypi.org/project/epilink/
2. Check that your package appears with the correct version
3. Test installation in a fresh environment:

```bash
# Create a test environment
python -m venv test-env
source test-env/bin/activate

# Install from TestPyPI (with dependencies from regular PyPI)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ epilink

# Test that it works
epilink --help
python -c "import epilink; print(epilink.__version__)"

# Clean up
deactivate
rm -rf test-env
```

---

## Option 2: Manual Upload (Quick Testing)

For quick local testing without GitHub Actions:

### 1. Build the package locally

```bash
# Install build tools
pip install --upgrade build twine

# Build distributions
python -m build

# This creates dist/epilink-X.Y.Z.tar.gz and dist/epilink-X.Y.Z-py3-none-any.whl
```

### 2. Check the build

```bash
twine check dist/*
```

### 3. Upload to TestPyPI

```bash
# You'll be prompted for your TestPyPI username and password/token
twine upload --repository testpypi dist/*

# Or specify credentials directly
twine upload --repository testpypi dist/* --username __token__ --password pypi-YOUR-TOKEN-HERE
```

### 4. Test installation

```bash
# Create a test environment
python -m venv test-env
source test-env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ epilink

# Test it
epilink --help

# Clean up
deactivate
rm -rf test-env
```

---

## Option 3: Using Tags for Test Releases

If you want to trigger test releases via git tags (similar to production releases):

### Modify test-release.yml

Change the trigger from `workflow_dispatch` to:

```yaml
on:
  push:
    tags:
      - "v*-rc*"    # Matches v1.0.0-rc1, v2.1.0-rc2, etc.
      - "v*-beta*"  # Matches v1.0.0-beta1, etc.
      - "v*-test*"  # Matches v1.0.0-test1, etc.
```

### Tag and push

```bash
# Create a test release tag
git tag v0.1.0-rc1
git push origin v0.1.0-rc1

# The workflow will automatically trigger
```

---

## Important Notes

### Version Numbers

- Your project uses `hatch-vcs` for dynamic versioning from git tags
- The version will be automatically determined from your git tags
- TestPyPI versions are separate from PyPI, so you can safely test

### TestPyPI Limitations

- Packages on TestPyPI are regularly deleted (every 30 days for old versions)
- It's for testing only, not for production use
- Some dependencies might not be available on TestPyPI

### Dependencies

When installing from TestPyPI, always use `--extra-index-url https://pypi.org/simple/` to pull dependencies from the main PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ epilink
```

### Cleanup

To remove old test distributions from your local `dist/` folder:

```bash
rm -rf dist/
```

---

## Troubleshooting

### "File already exists" error

TestPyPI (like PyPI) doesn't allow re-uploading the same version. Solutions:
- Delete your local `dist/` folder: `rm -rf dist/`
- Create a new git tag with a different version
- Use a version suffix (e.g., `-test2`, `-rc2`)

### Authentication errors

- For trusted publishing: Make sure the GitHub environment name matches exactly (`testpypi`)
- For API tokens: Ensure the username is `__token__` and password is your full token (starting with `pypi-`)

### Module not found after installation

- Make sure you used `--extra-index-url` to pull dependencies from main PyPI
- Check that all your dependencies are specified in `pyproject.toml`

---

## Ready for Production?

Once you've tested thoroughly on TestPyPI:

1. Create a production release tag: `git tag v1.0.0`
2. Push the tag: `git push origin v1.0.0`
3. The `release.yml` workflow will automatically publish to main PyPI

