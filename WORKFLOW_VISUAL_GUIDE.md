# TestPyPI Workflow - Visual Guide

## 🔄 The Problem (Before Fix)

```
Developer triggers workflow
           ↓
    No version tag exists
           ↓
    hatch-vcs generates: 0.0.0
           ↓
    Build package: epilink-0.0.0.tar.gz
           ↓
    Upload to TestPyPI
           ↓
    ❌ 400 Bad Request: "File already exists"
           ↓
    FAILURE - cannot test!
```

---

## ✅ The Solution (After Fix)

```
Developer triggers workflow
           ↓
    Create unique tag: v0.0.0.dev{run_number}
           ↓
    hatch-vcs detects tag
           ↓
    Build package: epilink-0.0.0.dev123.tar.gz
           ↓
    Upload to TestPyPI
           ↓
    ✅ SUCCESS - unique version uploaded!
           ↓
    Next run → dev124 → success again!
```

---

## 🎯 Version Lifecycle

```
Development Testing (TestPyPI)
┌─────────────────────────────┐
│ 0.0.0.dev1  ← First test    │
│ 0.0.0.dev2  ← Bug fix test  │
│ 0.0.0.dev3  ← Feature test  │
│ 0.0.0.dev4  ← Final test    │
└─────────────────────────────┘
           ↓
    All tests pass!
           ↓
Production Release (PyPI)
┌─────────────────────────────┐
│ 1.0.0       ← First release │
│ 1.1.0       ← Minor update  │
│ 2.0.0       ← Major version │
└─────────────────────────────┘
```

---

## 🔀 GitHub Actions Flow

```
┌──────────────────────────────────────────────┐
│ Developer clicks "Run workflow"              │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│ Job: build                                   │
│ ┌──────────────────────────────────────────┐ │
│ │ 1. Checkout code                         │ │
│ │ 2. Setup Python 3.11                     │ │
│ │ 3. Create tag: v0.0.0.dev{run_number}   │ │
│ │    (e.g., v0.0.0.dev42)                 │ │
│ │ 4. Install build tools                   │ │
│ │ 5. Build package → epilink-0.0.0.dev42  │ │
│ │ 6. Verify with twine check               │ │
│ │ 7. Upload artifacts                      │ │
│ └──────────────────────────────────────────┘ │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│ Job: publish-testpypi                        │
│ ┌──────────────────────────────────────────┐ │
│ │ 1. Download artifacts                    │ │
│ │ 2. Publish to TestPyPI                   │ │
│ │    ✅ SUCCESS!                           │ │
│ └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│ Package available at:                        │
│ https://test.pypi.org/project/epilink/       │
│                                              │
│ Install with:                                │
│ pip install --index-url                      │
│   https://test.pypi.org/simple/              │
│   --extra-index-url https://pypi.org/simple/ │
│   epilink                                    │
└──────────────────────────────────────────────┘
```

---

## 📊 Manual Release Flow

```
Developer runs: ./scripts/test_release.sh
           ↓
┌──────────────────────────────────────────────┐
│ 1. Create timestamp: 1709395200              │
│ 2. Create tag: v0.0.0.dev1709395200         │
│ 3. Build package                             │
│ 4. Clean up tag                              │
│ 5. Show instructions                         │
└──────────────────────┬───────────────────────┘
           ↓
Developer runs: twine upload --repository testpypi dist/*
           ↓
┌──────────────────────────────────────────────┐
│ Enter credentials (or use token)             │
└──────────────────────┬───────────────────────┘
           ↓
┌──────────────────────────────────────────────┐
│ ✅ Upload successful!                        │
│ View at: https://test.pypi.org/p/epilink/    │
└──────────────────────────────────────────────┘
```

---

## 🎨 Version Comparison

```
┌─────────────────┬──────────────────┬─────────────────┐
│ Version Type    │ Example          │ Use Case        │
├─────────────────┼──────────────────┼─────────────────┤
│ Dev (auto)      │ 0.0.0.dev1       │ GitHub Actions  │
│ Dev (timestamp) │ 0.0.0.dev170939  │ Manual builds   │
│ Alpha           │ 1.0.0a1          │ Alpha testing   │
│ Beta            │ 1.0.0b1          │ Beta testing    │
│ RC              │ 1.0.0rc1         │ Release cand.   │
│ Release         │ 1.0.0            │ Production      │
│ Post-release    │ 1.0.0.post1      │ Hotfix docs     │
└─────────────────┴──────────────────┴─────────────────┘

All are PEP 440 compliant ✅
```

---

## 🔑 Key Points

```
✅ DO:
  • Use GitHub Actions for test releases (easiest)
  • Let the workflow auto-generate versions
  • Test on TestPyPI before production
  • Use proper semantic versioning for production

❌ DON'T:
  • Manually create version tags for test releases
  • Try to upload the same version twice
  • Mix test versions with production tags
  • Skip testing on TestPyPI
```

---

## 🚀 Quick Start Commands

```bash
# Test on TestPyPI (GitHub Actions)
# → Just click "Run workflow" in GitHub UI

# Test on TestPyPI (Manual)
./scripts/test_release.sh
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    epilink

# Release to production PyPI
git tag v1.0.0
git push origin v1.0.0
# → Automatic via release.yml workflow
```

---

## 📚 Documentation Files

```
epilink/
├── TESTPYPI.md              ← Comprehensive guide
├── VERSIONING_QUICK_REF.md  ← Quick reference
├── RELEASE_CHECKLIST.md     ← Step-by-step checklist
├── .github/workflows/
│   ├── test-release.yml     ← Test releases (TestPyPI)
│   └── release.yml          ← Production releases (PyPI)
└── scripts/
    └── test_release.sh      ← Manual test helper
```

---

## 💡 Pro Tips

1. **GitHub Actions auto-increments**: Run as many times as needed!
2. **Manual builds use timestamps**: Each build is unique
3. **Clean up dist/ folder**: `rm -rf dist/` between builds
4. **Check TestPyPI**: Visit https://test.pypi.org/project/epilink/
5. **Test installation**: Always test from TestPyPI before production
6. **Production ready?**: Tag with `v1.0.0` for real release

---

## 🎯 Success Criteria

- ✅ GitHub Actions workflow completes without errors
- ✅ Package appears on TestPyPI with unique version
- ✅ Can install package from TestPyPI
- ✅ CLI command works: `epilink --help`
- ✅ Python import works: `import epilink`
- ✅ Ready to release to production PyPI

**Everything is set up and ready to go!** 🚀

