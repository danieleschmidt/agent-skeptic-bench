# CI/CD Workflow Documentation

This directory contains documentation for GitHub Actions workflows that should be implemented for the Agent Skeptic Bench project.

> **Note**: Due to security policies, workflow YAML files cannot be automatically created. Please manually implement these workflows in `.github/workflows/` based on the documentation below.

## Required Workflows

### 1. Tests and Quality (`test.yml`)

**Purpose**: Run comprehensive testing and code quality checks on every PR and push.

**Triggers**: 
- Pull requests to main branch
- Pushes to main branch
- Manual workflow dispatch

**Jobs**:
- **Unit Tests**: Run pytest with coverage reporting
- **Integration Tests**: Test full evaluation workflows
- **Code Quality**: Run ruff, black, isort, mypy
- **Security**: Run safety and bandit scans
- **Performance**: Basic performance regression tests

**Matrix Testing**:
- Python versions: 3.10, 3.11, 3.12
- OS: ubuntu-latest, macos-latest, windows-latest

### 2. Release (`release.yml`)

**Purpose**: Automated release process when version tags are pushed.

**Triggers**:
- Tags matching `v*` pattern (e.g., v1.0.0)

**Jobs**:
- Build and test package
- Generate changelog
- Create GitHub release
- Publish to PyPI
- Update documentation

### 3. Documentation (`docs.yml`)

**Purpose**: Build and deploy documentation to GitHub Pages.

**Triggers**:
- Pushes to main branch (docs changes)
- Manual workflow dispatch

**Jobs**:
- Build MkDocs documentation
- Deploy to GitHub Pages
- Update API documentation

### 4. Dependency Updates (`deps.yml`)

**Purpose**: Automated dependency updates and security patches.

**Triggers**:
- Schedule: Weekly on Sundays
- Manual workflow dispatch

**Jobs**:
- Run dependabot-like updates
- Test with new dependencies
- Create PRs for successful updates

## Workflow Implementation Guide

### Step 1: Create Workflow Files

Create the following files in `.github/workflows/`:

1. `test.yml` - Main testing workflow
2. `release.yml` - Release automation
3. `docs.yml` - Documentation deployment
4. `deps.yml` - Dependency management

### Step 2: Required Secrets

Configure these secrets in your GitHub repository:

- `PYPI_API_TOKEN` - For package publishing
- `CODECOV_TOKEN` - For coverage reporting (optional)

### Step 3: Branch Protection

Enable branch protection for `main`:
- Require status checks to pass
- Require branches to be up to date
- Require review from code owners
- Restrict pushes to specific people/teams

### Step 4: Security Configurations

Enable the following GitHub security features:
- Dependabot alerts
- Secret scanning
- Code scanning (CodeQL)

## Workflow Templates

See individual files in this directory for complete workflow implementations:

- `test-workflow-template.md` - Complete test workflow
- `release-workflow-template.md` - Release automation
- `docs-workflow-template.md` - Documentation deployment

## Integration Requirements

### External Services

The workflows integrate with:
- **PyPI**: Package publishing
- **Codecov**: Coverage reporting
- **GitHub Pages**: Documentation hosting

### Performance Considerations

- Use caching for pip dependencies
- Matrix builds run in parallel
- Skip unnecessary jobs on doc-only changes
- Use workflow conditionals to optimize runs

## Troubleshooting

Common issues and solutions:

### Test Failures
- Check Python version compatibility
- Verify all dependencies are pinned
- Review environment variable requirements

### Release Issues
- Ensure version tags match semver format
- Verify PyPI token has correct permissions
- Check that all tests pass before tagging

### Documentation Deployment
- Verify GitHub Pages is enabled
- Check MkDocs configuration
- Ensure all documentation links are valid

For additional help, see the GitHub Actions documentation or create an issue.