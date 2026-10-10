# Security Policy

## Supported versions

Only the latest release on `main` receives fixes.

## Reporting a vulnerability

Please **do not open a public issue**. Report privately through
[GitHub Security Advisories](https://github.com/KrishnaVaibhav/TrachSplitter/security/advisories/new).

Include what you found, how to reproduce it, and the impact you expect. You'll get an
acknowledgement within 7 days, and we'll agree on a disclosure date once a fix is ready.

## Scope

TrachSplitter runs fully offline except for downloading Demucs model weights from the
official Meta/PyTorch hosts on first use. Relevant reports include, for example: crafted
audio files that cause code execution, unsafe file writes outside the chosen folder, or
tampered model downloads.
