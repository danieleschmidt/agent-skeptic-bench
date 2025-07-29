# Test Workflow Template

Create this file as `.github/workflows/test.yml`:

```yaml
name: Tests and Quality

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.10', '3.11', '3.12']
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
          
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
        
    - name: Run ruff linting
      run: ruff check src tests
      
    - name: Check code formatting with black
      run: black --check src tests
      
    - name: Check import sorting with isort
      run: isort --check-only src tests
      
    - name: Type checking with mypy
      run: mypy src
      
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=agent_skeptic_bench --cov-report=xml --cov-report=term
        
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v --cov=agent_skeptic_bench --cov-append --cov-report=xml
        
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
        python-version: '3.11'
        
    - name: Install security tools
      run: |
        pip install safety bandit[toml]
        
    - name: Run safety check
      run: safety check
      
    - name: Run bandit security linting
      run: bandit -r src/ -f json -o bandit-report.json
      
    - name: Upload bandit results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: bandit-results
        path: bandit-report.json
```

## Key Features

- **Matrix Testing**: Tests across Python 3.10-3.12 and major OS platforms
- **Caching**: Speeds up builds by caching pip dependencies
- **Code Quality**: Runs ruff, black, isort, and mypy
- **Security**: Includes safety and bandit security scans
- **Coverage**: Generates and uploads coverage reports to Codecov
- **Parallel Jobs**: Security scans run separately for faster feedback

## Customization Options

### Skip Matrix on Draft PRs
Add this condition to the test job:
```yaml
if: github.event.pull_request.draft == false
```

### Add Performance Tests
Add a separate job:
```yaml
performance:
  runs-on: ubuntu-latest
  steps:
    # ... setup steps ...
    - name: Run performance benchmarks
      run: pytest tests/performance/ --benchmark-only
```

### Environment Variables
Add environment variables for API keys (use repository secrets):
```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```