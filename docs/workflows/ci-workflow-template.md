# GitHub Actions CI/CD Workflow Templates

This document provides complete GitHub Actions workflow templates for the Agent Skeptic Bench project. Copy these into `.github/workflows/` directory.

## Required Workflows for Production Readiness

### 1. Main CI/CD Pipeline (`ci.yml`)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly dependency check

env:
  PYTHON_VERSION: '3.10'
  REGISTRY_IMAGE: agent-skeptic-bench

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Lint with ruff
      run: ruff check src/ tests/
    
    - name: Format check with black
      run: black --check src/ tests/
    
    - name: Type check with mypy
      run: mypy src/
    
    - name: Test with pytest
      run: |
        pytest --cov=agent_skeptic_bench --cov-report=xml --cov-report=term
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install security tools
      run: |
        pip install bandit safety semgrep
    
    - name: Run Bandit security linter
      run: bandit -r src/ -f json -o bandit-report.json
    
    - name: Check dependencies for vulnerabilities
      run: safety check --json --output safety-report.json
    
    - name: Run Semgrep
      run: semgrep --config=auto src/

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install build tools
      run: |
        python -m pip install --upgrade pip
        pip install build hatch
    
    - name: Build package
      run: python -m build
    
    - name: Store distribution packages
      uses: actions/upload-artifact@v3
      with:
        name: python-package-distributions
        path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')
    environment:
      name: pypi
      url: https://pypi.org/p/agent-skeptic-bench
    permissions:
      id-token: write
    
    steps:
    - name: Download distribution packages
      uses: actions/download-artifact@v3
      with:
        name: python-package-distributions
        path: dist/
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
```

### 2. Security Scanning (`security.yml`)

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * *'  # Daily security scan

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python

    - name: Autobuild
      uses: github/codeql-action/autobuild@v2

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
    - name: Dependency Review
      uses: actions/dependency-review-action@v3

  trivy:
    name: Trivy Security Scan
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
```

### 3. Performance Testing (`performance.yml`)

```yaml
name: Performance Testing

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 4 * * 0'  # Weekly performance baseline

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev,test]"
        pip install pytest-benchmark
    
    - name: Run performance benchmarks
      run: |
        pytest tests/performance/ --benchmark-json=benchmark.json
    
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

### 4. Container Build (`docker.yml`)

```yaml
name: Container Build & Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      security-events: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
```

## Setup Instructions

1. Create `.github/workflows/` directory in your repository
2. Copy the above workflows into separate `.yml` files
3. Configure repository secrets:
   - `CODECOV_TOKEN`: For code coverage reporting
   - `PYPI_API_TOKEN`: For package publishing
4. Enable GitHub Advanced Security features:
   - Dependency graph
   - Security advisories
   - Code scanning alerts
5. Configure branch protection rules requiring status checks

## Required Repository Settings

- Enable "Require status checks to pass before merging"
- Enable "Require branches to be up to date before merging"
- Enable vulnerability alerts
- Enable security updates
- Configure CODEOWNERS file for automated review assignments

## Next Steps

After implementing these workflows:
1. Monitor initial runs and fix any configuration issues
2. Set up notification preferences for failed builds
3. Configure deployment environments for staged releases
4. Implement additional environment-specific testing