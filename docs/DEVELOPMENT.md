# Development Guide

This guide helps you set up a development environment and contribute to Agent Skeptic Bench.

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/agent-skeptic-bench.git
cd agent-skeptic-bench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

## Development Environment

### Python Version
- **Minimum**: Python 3.10
- **Recommended**: Python 3.11 or 3.12
- **Testing**: We test against 3.10, 3.11, and 3.12

### IDE Configuration

#### VS Code
Install these extensions:
- Python (Microsoft)
- Black Formatter
- isort
- Ruff

Add this to `.vscode/settings.json`:
```json
{
    "python.defaultInterpreter": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm
1. Set interpreter to your venv Python
2. Configure code style to use Black
3. Enable ruff for linting
4. Set isort to use black profile

### Environment Variables

Create a `.env` file for development (never commit this):
```bash
# AI Provider API Keys (for testing)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_AI_API_KEY=your_key_here

# Development settings
AGENT_SKEPTIC_DEBUG=true
PYTEST_CURRENT_TEST=true
```

## Code Quality

### Pre-commit Hooks
Our pre-commit configuration runs:
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with newline
- **check-yaml**: Validates YAML files
- **black**: Code formatting
- **isort**: Import sorting
- **ruff**: Linting and code analysis
- **mypy**: Type checking

### Manual Quality Checks
```bash
# Format code
black src tests
isort src tests

# Lint code
ruff check src tests

# Type checking
mypy src

# Run all quality checks
pre-commit run --all-files
```

## Testing

### Test Structure
```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_agents.py
│   ├── test_benchmark.py
│   └── test_scenarios.py
├── integration/             # Integration tests (slower, external dependencies)
│   ├── test_full_evaluation.py
│   └── test_model_integration.py
└── performance/             # Performance benchmarks
    └── test_benchmark_speed.py
```

### Running Tests
```bash
# Run all tests
pytest

# Run only unit tests (fast)
pytest tests/unit/

# Run with coverage
pytest --cov=agent_skeptic_bench --cov-report=html

# Run specific test
pytest tests/unit/test_agents.py::test_create_skeptic_agent

# Run tests in parallel (if you have pytest-xdist)
pytest -n auto
```

### Writing Tests

#### Unit Test Example
```python
import pytest
from agent_skeptic_bench import create_skeptic_agent

def test_create_skeptic_agent():
    """Test that skeptic agent creation works with default parameters."""
    agent = create_skeptic_agent(model="mock", skepticism_level="calibrated")
    
    assert agent is not None
    assert agent.skepticism_level == "calibrated"
    assert agent.model == "mock"

@pytest.mark.parametrize("level", ["low", "calibrated", "high"])
def test_skepticism_levels(level):
    """Test all supported skepticism levels."""
    agent = create_skeptic_agent(model="mock", skepticism_level=level)
    assert agent.skepticism_level == level
```

#### Integration Test Example
```python
import pytest
from agent_skeptic_bench import SkepticBenchmark, run_full_evaluation

@pytest.mark.integration
@pytest.mark.slow
def test_full_evaluation_workflow():
    """Test complete evaluation workflow with mock agents."""
    benchmark = SkepticBenchmark()
    
    # Use mock agents to avoid API calls
    results = run_full_evaluation(
        skeptic_agent=create_mock_skeptic(),
        categories=["factual_claims"],
        max_scenarios=5  # Limit for speed
    )
    
    assert len(results.scenario_results) == 5
    assert results.overall_score is not None
```

### Test Fixtures
Common fixtures are defined in `tests/conftest.py`:

```python
@pytest.fixture
def mock_skeptic_agent():
    """Create a mock skeptic agent for testing."""
    return create_skeptic_agent(model="mock", skepticism_level="calibrated")

@pytest.fixture
def sample_scenario():
    """Create a sample scenario for testing."""
    return Scenario(
        name="test_scenario",
        category="factual_claims",
        adversary_claim="The earth is flat",
        correct_skepticism_level=0.95
    )
```

## Architecture

### Key Components

#### 1. Benchmark (`src/agent_skeptic_bench/benchmark.py`)
- Core benchmark orchestration
- Scenario loading and management
- Result aggregation

#### 2. Agents (`src/agent_skeptic_bench/agents.py`)
- Skeptic agent implementations
- Adversarial agent interfaces
- Agent factory functions

#### 3. Scenarios (`src/agent_skeptic_bench/scenarios/`)
- Individual scenario definitions
- Category-based organization
- Scenario validation

#### 4. Evaluation (`src/agent_skeptic_bench/evaluation.py`)
- Evaluation metrics and scoring
- Result analysis and reporting
- Performance measurement

### Adding New Features

#### New Scenario Category
1. Create directory: `src/agent_skeptic_bench/scenarios/new_category/`
2. Add scenario files with proper metadata
3. Update scenario loader in `benchmark.py`
4. Add tests in `tests/unit/test_scenarios.py`

#### New Evaluation Metric
1. Add metric class to `src/agent_skeptic_bench/metrics/`
2. Implement required interface methods
3. Update evaluation pipeline
4. Add comprehensive tests

#### New Agent Type
1. Add agent class to `src/agent_skeptic_bench/agents/`
2. Update factory function
3. Add integration tests
4. Update documentation

## Contributing

### Workflow
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes with tests
4. Run quality checks: `pre-commit run --all-files`
5. Run test suite: `pytest`
6. Commit with conventional commits format
7. Push and create pull request

### Commit Message Format
We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

feat(agents): add new GPT-4 skeptic agent implementation
fix(benchmark): resolve scenario loading race condition
docs(readme): update installation instructions
test(integration): add comprehensive API tests
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`

### Pull Request Guidelines
- Fill out the PR template completely
- Ensure all tests pass
- Add tests for new functionality
- Update documentation as needed
- Request review from maintainers

## Performance Considerations

### Optimization Tips
- Use async/await for concurrent evaluations
- Cache expensive computations
- Batch API calls when possible
- Profile with `pytest-benchmark` for critical paths

### Memory Management
- Be mindful of large scenario datasets
- Use generators for streaming evaluations
- Clean up resources in tests

## Debugging

### Common Issues

#### Test Failures
```bash
# Run with verbose output
pytest -v -s

# Drop into debugger on failure
pytest --pdb

# Only run failed tests
pytest --lf
```

#### Import Errors
```bash
# Verify installation
pip show agent-skeptic-bench

# Reinstall in development mode
pip install -e ".[dev]"
```

#### API Rate Limits
- Use mock agents in tests
- Implement exponential backoff
- Consider using smaller test datasets

### Debug Configuration
Add this to your `.env`:
```bash
AGENT_SKEPTIC_DEBUG=true
PYTEST_CURRENT_TEST=true
LOG_LEVEL=DEBUG
```

## Documentation

### Building Docs Locally
```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve docs locally
mkdocs serve

# Build static docs
mkdocs build
```

### Documentation Structure
```
docs/
├── index.md              # Main documentation
├── api/                  # API reference
├── guides/               # User guides
├── development/          # Development docs
└── workflows/            # CI/CD documentation
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. After merge, tag release: `git tag v1.0.0`
5. Push tag: `git push origin v1.0.0`
6. GitHub Actions will handle PyPI release

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and community support
- **Discord**: Real-time chat and collaboration
- **Email**: Direct contact for sensitive issues

Happy coding! 🚀