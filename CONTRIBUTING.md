# Contributing to TrachSplitter

Thanks for helping! This page covers how changes get into `main`.

## Ground rules

- **`main` is protected.** Nobody pushes to it directly. Every change goes through a pull request.
- **CI must be green.** A PR can only merge when the **CI passed** check succeeds: lint, tests on Linux and Windows, and a full Windows app build.
- **One approving review** from a code owner (see `.github/CODEOWNERS`) is required. New commits after approval need a fresh review.
- **All review conversations must be resolved** before merging.
- **Linear history:** PRs are squash-merged; the PR title becomes the commit message.
- Be kind and constructive: see the [Code of Conduct](CODE_OF_CONDUCT.md).

## Workflow

1. Open an issue first for anything bigger than a small fix, so we can agree on the approach.
2. Fork the repo (or create a branch if you have access): `git checkout -b fix/short-description`.
3. Make your change with tests.
4. Run the checks locally (below).
5. Open a PR against `main` and fill in the template.

### PR titles

Use [Conventional Commits](https://www.conventionalcommits.org/) style. CI checks this:

```
feat: add stem fade handles
fix: keep mixer in sync after cancelling separation
docs: explain LUFS targets
```

Allowed types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`, `style`, `revert`.

## Development setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python main.py
```

For a CPU-only setup (smaller download, what CI uses):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
```

## Checks to run before pushing

```bash
ruff check .            # lint (real bugs only: undefined names, unused imports…)
python -m pytest        # unit + CLI + offscreen GUI tests
python build.py --no-package   # optional: full app build (several minutes)
```

## Code guidelines

- **Never block the UI thread.** Separation, rendering, analysis and file writes run in workers (`_run_task`, `QThread`). Worker code must not touch widgets.
- **Audio arrays are `(channels, samples)` float32 at 44.1 kHz** throughout.
- **Write audio with `audio_editor.write_audio_file` / `AudioEditor.export_audio`.** They write in blocks (a single large OGG write crashes libsndfile).
- **UI colours come from `ui_theme.py` tokens.** Don't hard-code hex values in `gui.py`; check contrast (≥ 4.5:1 for text) in both light and dark.
- **Results go to `self.notify(...)`**, not modal message boxes. Keep `QMessageBox` for errors and confirmations.
- Add a test for every bug fix that reproduces the bug.

## Reporting bugs and security issues

- Bugs and ideas: [open an issue](https://github.com/KrishnaVaibhav/TrachSplitter/issues/new/choose).
- Security problems: report privately, see [SECURITY.md](SECURITY.md).
