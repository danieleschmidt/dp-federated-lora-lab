# Contributing to DP-Federated LoRA Lab

We welcome contributions! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes
5. Run tests: `pytest`
6. Submit a pull request

## 📋 Development Setup

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- Git

### Installation
```bash
# Clone your fork
git clone https://github.com/yourusername/dp-federated-lora-lab.git
cd dp-federated-lora-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m privacy     # Privacy-related tests only
pytest -m integration # Integration tests only
```

### Writing Tests
- Place tests in the `tests/` directory
- Follow the naming convention: `test_*.py`
- Use descriptive test names
- Include privacy and security test cases
- Test edge cases and error conditions

## 🎨 Code Style

We use automated code formatting and linting:

### Formatting
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

### Standards
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Maximum line length: 88 characters
- Use descriptive variable and function names
- Add docstrings for public APIs

## 🔒 Security Guidelines

Since this project deals with privacy and security:

1. **Never commit sensitive data** (API keys, private keys, real datasets)
2. **Review privacy implications** of all changes
3. **Test differential privacy guarantees** thoroughly
4. **Use secure coding practices** (input validation, error handling)
5. **Run security checks**: `bandit src/` and `safety check`

## 📖 Documentation

### Code Documentation
- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include examples in docstrings where helpful
- Document privacy parameters and guarantees

### Project Documentation
- Update README.md for user-facing changes
- Add tutorials for new features
- Update API documentation

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information** (Python version, OS, GPU)
2. **Minimal reproduction case**
3. **Expected vs actual behavior**
4. **Privacy configuration** (if relevant)
5. **Error messages and stack traces**

Use our bug report template: [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md)

## ✨ Feature Requests

For new features:

1. **Check existing issues** to avoid duplicates
2. **Explain the use case** and motivation
3. **Consider privacy implications**
4. **Propose an API design** (if applicable)
5. **Estimate complexity** and testing requirements

Use our feature request template: [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md)

## 🔄 Pull Request Process

### Before Submitting
1. **Create an issue** to discuss major changes
2. **Write tests** for your changes
3. **Update documentation** as needed
4. **Run the full test suite**
5. **Check code style** and security

### PR Requirements
- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`black`, `isort`)
- [ ] Linting passes (`flake8`, `ruff`)
- [ ] Type checking passes (`mypy`)
- [ ] Security checks pass (`bandit`)
- [ ] Documentation updated
- [ ] Privacy guarantees maintained

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Privacy enhancement
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Privacy tests verify guarantees

## Privacy Impact
Describe any changes to privacy guarantees or parameters

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🏷️ Release Process

For maintainers:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. Publish to PyPI (automated)

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help newcomers get started
- Share knowledge and best practices
- Focus on constructive feedback
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## 📧 Contact

- **Issues**: Use GitHub Issues for bugs and features
- **Discussions**: Use GitHub Discussions for questions
- **Security**: Email security@terragonlabs.com for vulnerabilities
- **Maintainers**: @danielschmidt (Daniel Schmidt)

## 🎯 Priority Areas

We especially welcome contributions in:

- **Privacy mechanisms**: New DP algorithms, privacy accounting
- **LoRA variants**: Efficient adaptation methods
- **Federation protocols**: Cross-silo support, Byzantine robustness  
- **Benchmarking**: Real-world scenarios, evaluation metrics
- **Documentation**: Tutorials, examples, best practices

Thank you for contributing to privacy-preserving federated learning! 🛡️🤖