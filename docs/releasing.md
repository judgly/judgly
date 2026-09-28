# Releasing judgly

Releases are built and uploaded by GitHub Actions (`.github/workflows/release.yml`). PyPI
accepts the upload through [trusted publishing](https://docs.pypi.org/trusted-publishers/):
GitHub hands the job a short-lived OpenID Connect token and PyPI checks it against the
publisher you register once. No PyPI API token is stored in the repository or its secrets.

What gets built: the sdist (with the llama.cpp sources needed to build libjudgly), about 12 MB,
and one wheel, `judgly-<version>-py3-none-macosx_14_0_arm64.whl`, about 3.3 MB (15 MB unpacked),
for Apple silicon on macOS 14 or later. The wheel carries libjudgly, both packs' heads and
calibration records, the packs' LICENSE files, LICENSE, NOTICE and LICENSES/. There are no Linux,
Windows or Intel Mac wheels yet; on those platforms pip would try to build from the sdist.

The wheel's platform tag comes only from the `MACOSX_DEPLOYMENT_TARGET` environment variable
(scikit-build-core has no setting for it). CI and `release.yml` set it to `14.0`. Locally, always
build with `MACOSX_DEPLOYMENT_TARGET=14.0 uv build`: a plain `uv build` tags the wheel with the
build machine's macOS version (for example `macosx_27_0_arm64`), and pip on older macOS would
refuse it. The library inside is built for macOS 14.0 either way (CMakeLists.txt sets that
default).

## Release procedure

The steps below release version `X.Y.Z` (for the first functional release, 0.1.0) from the
repository `judgly/judgly`. Do them in order: steps 3 and 4 must be done before any tag is
pushed, because the release workflow publishes from a tag.

### 1. Review what will be published

On a clean checkout of `main`, check that the version is `X.Y.Z` in `pyproject.toml` and
`CITATION.cff` (with `date-released`), that the CHANGELOG entry for `X.Y.Z` reads right, and
that the README and docs show the numbers in `docs/results/`.

### 2. Run the local checks (the ones CI runs, plus the model tests)

```sh
uvx ruff check .
uv run --no-project python scripts/check_licenses.py
make check                                   # contamination checker on data/tiers
make verify-data                             # tiers equal data/tiers.sha256
JUDGLY_MODEL_DIR=/path/to/models uv run --group pipeline pytest -q -rs
JUDGLY_MODEL_DIR=/path/to/models uv run --group pipeline python docs/tools/check_docs.py
rm -rf dist && MACOSX_DEPLOYMENT_TARGET=14.0 uv build
uvx twine check --strict dist/*
uv run --isolated --no-project --with dist/judgly-X.Y.Z-py3-none-macosx_14_0_arm64.whl \
    python -c "import judgly; print(judgly.__version__, judgly.native_version())"
```

All must pass; the last line must print `X.Y.Z` and `"judgly":"X.Y.Z"`.

### 3. Protect the `pypi` environment (once, before any tag)

In the GitHub repository settings, under Environments, create an environment named `pypi`.
Add yourself as a required reviewer, and under deployment branches and tags allow only tags
matching `v*`. The `publish` job of `release.yml` runs in this environment, so every upload
waits for your approval.

### 4. Register the PyPI trusted publisher (once, before any tag)

On pypi.org, open the `judgly` project, Manage → Publishing, and add a GitHub publisher with
owner `judgly`, repository `judgly`, workflow `release.yml` and environment `pypi`. No API
token is needed or stored.

### 5. Check CI and a build-only release run

Push `main` and check that the first CI run (`ci.yml`, on `macos-15`) passes. Then start
`release.yml` by hand (Actions → release → Run workflow): a manual run builds and checks the
sdist and wheel but does not publish. Download its `dist` artifact and check that the wheel name
ends in `macosx_14_0_arm64.whl`.

### 6. Optional: Zenodo DOI (before tagging)

Zenodo archives a GitHub *release* (not a bare tag), so switch it on first: log in at zenodo.org
with GitHub, open Account → GitHub, press "Sync now" and enable `judgly/judgly`. Zenodo reads
`CITATION.cff` for the record's metadata.

### 7. Tag the release

The tag must be `v` plus the version exactly, or the build job stops.

```sh
git tag -a vX.Y.Z -m "judgly X.Y.Z"
git push origin vX.Y.Z
```

The `release` workflow builds and checks the sdist and wheel, then the `publish` job waits for
approval in the `pypi` environment. Look at the build log and the `dist` artifact, then
approve. Check `https://pypi.org/project/judgly/X.Y.Z/` and, in a clean environment,
`uv run --isolated --no-project --with judgly==X.Y.Z python -c "import judgly; print(judgly.native_version())"`.
The README on PyPI uses absolute links to the `vX.Y.Z` tag, so its images and links resolve
once the tag is pushed.

### 8. GitHub release and social preview

Create a GitHub release for the tag with the CHANGELOG entry as its notes (with Zenodo switched
on, this release is what it archives). Once, upload `docs/assets/social-preview.png` (1280 x
640) under Settings → General → Social preview.

### 9. Record the DOI (if you minted one)

Zenodo shows a DOI for all versions (the "concept" DOI) and one per version. Put the concept DOI
in `CITATION.cff` as `doi:` and commit it to `main`; the next release carries it.

## If something is wrong

PyPI never accepts the same file name twice, even after deletion, so a broken X.Y.Z is replaced by
X.Y.(Z+1), never re-uploaded.

- **Before approval.** Reject the `publish` job in the `pypi` environment. Delete the tag
  (`git push origin :refs/tags/vX.Y.Z && git tag -d vX.Y.Z`), fix, and tag again.
- **After upload: yank.** On pypi.org, Your projects → judgly → Manage → Releases → X.Y.Z →
  Options → Yank, and give the reason. A yanked release stays downloadable for anyone who pins
  `judgly==X.Y.Z` exactly, but installers no longer pick it for `judgly` or a version range. Then
  fix, bump to X.Y.(Z+1) (pyproject.toml, CITATION.cff, a CHANGELOG entry that says what was wrong)
  and release again from step 1.
- **Delete** the release on PyPI only if it must not be downloadable at all (for example a
  licence problem in a shipped file); the version number stays used up.
- Mark the GitHub release as a pre-release or edit its notes to point at the fixed version.
  A Zenodo record cannot be deleted; publish the fixed version as a new Zenodo version.

## Later releases

1. Set the version in `pyproject.toml` (`[project] version`; the C library and
   `judgly.__version__` read it from there), set the same `version` and the release date as
   `date-released` in `CITATION.cff`, move the CHANGELOG's Unreleased entries under the new
   version, and commit.
2. Steps 1, 2 and 5 to 9 above.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and on pull requests, on a macOS arm64
runner: ruff, the licence check, the sdist and wheel build, an import of the wheel in a clean
environment, the contamination-checker tests and the rest of the test suite. No model file is
available there, so the tests marked `model` skip; run them locally (step 2) before a release.
