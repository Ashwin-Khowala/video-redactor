# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-06-13

### Added
- Refactored monolithic script into a clean, structured, and modular Python package.
- Implemented robust pattern matching (`MatchConfig`) for emails, phone numbers, and custom keywords.
- Implemented robust frame boundary checking and Gaussian blur logic (`apply_blur`).
- Integrated dynamic frame skipping and static/scene transition frame skips to maximize execution speed.
- Provided multi-worker parallel chunking using FFmpeg copy segmenting and audio recovery.
- Added test coverage with pytest for patterns, blurring, and process configurations.
- Added GitHub Actions workflow configuration for CI (linting, testing) and PyPI release.
