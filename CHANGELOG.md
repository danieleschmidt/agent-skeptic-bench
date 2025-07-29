# Changelog

All notable changes to Agent Skeptic Bench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive SDLC enhancement with security configurations
- GitHub issue templates for bug reports, features, and scenarios
- Development environment setup with VS Code and devcontainer support
- CI/CD workflow documentation and templates
- Enhanced security policy and vulnerability reporting process
- Developer documentation and contribution guidelines

### Changed
- Updated project structure to support enhanced development workflow

### Security
- Added comprehensive security policy (SECURITY.md)
- Documented vulnerability reporting process
- Added security scanning recommendations for CI/CD

## [1.0.0] - 2025-01-XX

### Added
- Initial release of Agent Skeptic Bench
- Core benchmark framework for testing AI agent skepticism
- Support for multiple LLM providers (OpenAI, Anthropic, Google)
- Comprehensive scenario categories:
  - Factual claims evaluation
  - Flawed plan detection
  - Persuasion attack resistance
  - Evidence quality assessment
  - Epistemic calibration testing
- Built-in adversarial agents with different manipulation strategies
- Evaluation metrics for skepticism appropriateness
- Command-line interface for benchmark execution
- Python API for programmatic usage
- Comprehensive test suite with unit and integration tests
- Pre-commit hooks for code quality
- Documentation and examples

### Developer Experience
- Comprehensive README with usage examples
- Type hints and mypy support
- Black code formatting
- Ruff linting and code analysis
- pytest testing framework with coverage
- Pre-commit hooks for quality assurance

## Release Notes Template

### [X.Y.Z] - YYYY-MM-DD

#### Added
- New features and capabilities

#### Changed
- Changes to existing functionality

#### Deprecated
- Features that will be removed in future versions

#### Removed
- Features removed in this version

#### Fixed
- Bug fixes and corrections

#### Security
- Security-related changes and fixes

---

## Release Process

1. Update version number in `pyproject.toml`
2. Update this CHANGELOG.md with new version details
3. Create pull request with version changes
4. After merge, create and push git tag: `git tag vX.Y.Z`
5. GitHub Actions will automatically create release and publish to PyPI

## Version Schema

We use [Semantic Versioning](https://semver.org/):
- **MAJOR** (X): Incompatible API changes
- **MINOR** (Y): Backward-compatible functionality additions
- **PATCH** (Z): Backward-compatible bug fixes

### Pre-release Versions
- **Alpha** (X.Y.Z-alpha.N): Early development, unstable
- **Beta** (X.Y.Z-beta.N): Feature complete, testing phase
- **RC** (X.Y.Z-rc.N): Release candidate, final testing