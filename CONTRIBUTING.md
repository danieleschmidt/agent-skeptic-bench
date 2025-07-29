# Contributing to Agent Skeptic Bench

Thank you for your interest in contributing! This document provides guidelines for contributing to the Agent Skeptic Bench project.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/yourusername/agent-skeptic-bench/issues) page
- Search existing issues before creating a new one
- Include clear reproduction steps and environment details
- Use appropriate issue templates

### Contributing Code

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Install development dependencies**: `pip install -e ".[dev]"`
4. **Set up pre-commit hooks**: `pre-commit install`
5. **Make your changes** following our coding standards
6. **Add tests** for new functionality
7. **Run the test suite**: `pytest`
8. **Commit your changes**: Use clear, descriptive commit messages
9. **Push to your fork**: `git push origin feature/your-feature-name`
10. **Create a Pull Request**: Use our PR template

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/agent-skeptic-bench.git
cd agent-skeptic-bench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

### Coding Standards

- **Python 3.10+** required
- **Code formatting**: Use `black` and `isort` (configured in `pyproject.toml`)
- **Linting**: Use `ruff` for fast Python linting
- **Type hints**: Use `mypy` for static type checking
- **Documentation**: Use Google-style docstrings
- **Line length**: Maximum 88 characters

### Testing Guidelines

- Write tests for all new functionality
- Maintain or improve code coverage
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Use pytest fixtures for setup

```python
def test_skeptic_agent_identifies_false_claim():
    # Arrange
    agent = create_skeptic_agent(model="test")
    false_claim = "The Earth is flat with overwhelming evidence"
    
    # Act
    response = agent.evaluate_claim(false_claim)
    
    # Assert
    assert response.skepticism_level > 0.8
    assert "evidence" in response.evidence_requests.lower()
```

### Adding New Scenarios

We welcome new adversarial scenarios! Please:

1. **Follow the scenario template** in `scenarios/template.py`
2. **Include comprehensive metadata**
3. **Provide clear evaluation criteria**
4. **Add appropriate test cases**

Example scenario structure:
```python
from agent_skeptic_bench import Scenario, AdversaryAgent

class YourAdversaryAgent(AdversaryAgent):
    def generate_claim(self):
        return "Your false but plausible claim here"
    
    def respond_to_skepticism(self, skeptic_response):
        # Implementation of how adversary responds to challenges
        pass

scenario = Scenario(
    name="your_scenario_name",
    category="factual_claims",  # or appropriate category
    adversary=YourAdversaryAgent(),
    correct_skepticism_level=0.8,  # Expected skepticism level
    evaluation_criteria={
        "evidence_requests": ["peer_review", "replication"],
        "red_flags": ["anecdotal", "unfalsifiable"]
    }
)
```

### Documentation

- Update relevant documentation for new features
- Use clear, concise language
- Include code examples where helpful
- Update the README if adding major features

### Performance Considerations

- Profile performance-critical code
- Consider memory usage for large datasets
- Use async/await for I/O operations where appropriate
- Optimize evaluation loops for benchmark efficiency

## Pull Request Process

1. **Ensure all tests pass**: `pytest`
2. **Run code quality checks**: `pre-commit run --all-files`
3. **Update documentation** as needed
4. **Add changelog entry** if applicable
5. **Fill out the PR template** completely
6. **Link related issues** using keywords (fixes #123)

### PR Review Process

- All PRs require at least one review
- Address review feedback promptly
- Keep PRs focused and reasonably sized
- Ensure CI checks pass before requesting review

## Types of Contributions

### High Priority
- New adversarial scenarios
- Improved evaluation metrics
- Better agent architectures
- Performance optimizations
- Bug fixes

### Welcome Contributions
- Documentation improvements
- Example notebooks
- Integration with new models
- Visualization tools
- Analysis utilities

### Community Guidelines

- Be respectful and constructive
- Focus on the contribution, not the contributor
- Ask questions when unclear
- Help others learn and improve
- Share knowledge and best practices

## Getting Help

- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bug reports and feature requests
- **Discord**: Join our [community chat](https://discord.gg/skeptic-bench)
- **Email**: skeptic-bench@yourdomain.com for private matters

## Recognition

Contributors are recognized through:
- GitHub contributor graphs
- Release notes acknowledgments
- Project documentation credits
- Community highlights

Thank you for helping make AI agents more epistemically humble and truthful!