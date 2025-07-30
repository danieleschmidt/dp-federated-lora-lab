# CI/CD Workflow Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) workflows for the DP-Federated LoRA Lab project.

> **Note**: This documentation describes required GitHub Actions workflows. The actual YAML files should be created by repository maintainers in `.github/workflows/`.

## 🔄 Overview

Our CI/CD pipeline ensures code quality, security, and privacy guarantees through automated testing and deployment processes.

### Workflow Triggers
- **Push** to `main` branch
- **Pull Request** creation and updates
- **Release** tag creation
- **Schedule** for nightly builds and security scans

## 🧪 Continuous Integration

### Main CI Workflow

**File**: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Mondays

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.9", "3.10", "3.11"]
        
  lint-and-format:
    # Code quality checks
    
  security-scan:
    # Security and vulnerability scanning
    
  privacy-tests:
    # Privacy-specific validation
    
  documentation:
    # Build and validate documentation
```

### Test Matrix

| OS | Python | PyTorch | CUDA | Purpose |
|----|--------|---------|------|---------|
| Ubuntu 20.04 | 3.9 | 2.0 | 11.8 | Primary |
| Ubuntu 22.04 | 3.10 | 2.1 | 12.1 | Latest |
| Windows | 3.11 | 2.0 | - | Compatibility |
| macOS | 3.10 | 2.0 | - | Apple Silicon |

### Required Checks

All PRs must pass these checks:

1. **Code Quality**
   ```bash
   black --check src/ tests/
   isort --check-only src/ tests/
   flake8 src/ tests/
   ruff check src/ tests/
   ```

2. **Type Checking**
   ```bash
   mypy src/
   ```

3. **Security Scanning**
   ```bash
   bandit -r src/
   safety check
   pip-audit
   ```

4. **Testing**
   ```bash
   pytest --cov=src --cov-report=xml
   pytest -m privacy  # Privacy-specific tests
   ```

5. **Privacy Validation**
   ```bash
   python -m dp_federated_lora.audit.privacy_checker
   ```

## 🔒 Security Workflows

### Security Scanning

**File**: `.github/workflows/security.yml`

```yaml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 3 * * *'  # Daily at 3 AM

jobs:
  codeql:
    # GitHub CodeQL analysis
    
  dependency-check:
    # OWASP dependency check
    
  container-scan:
    # Docker image vulnerability scan
    
  secrets-scan:
    # Secrets detection
```

### Security Gates

- **Critical vulnerabilities**: Block deployment
- **High vulnerabilities**: Require review
- **Medium/Low vulnerabilities**: Create issues
- **Secrets detected**: Block immediately

## 📋 Privacy Testing

### Privacy Validation Workflow

**File**: `.github/workflows/privacy.yml`

```yaml
name: Privacy Validation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  dp-guarantee-tests:
    # Test differential privacy guarantees
    
  privacy-accounting:
    # Validate privacy budget accounting
    
  membership-inference:
    # Test against membership inference attacks
    
  gradient-inversion:
    # Test gradient inversion resistance
```

### Privacy Test Requirements

1. **Differential Privacy Bounds**
   ```python
   def test_epsilon_delta_bounds():
       # Verify (ε, δ)-DP guarantees
       assert privacy_engine.get_epsilon(delta=1e-5) <= max_epsilon
   ```

2. **Composition Theorems**
   ```python
   def test_privacy_composition():
       # Test advanced composition vs basic composition
       assert advanced_epsilon < basic_epsilon
   ```

3. **Privacy Amplification**
   ```python
   def test_subsampling_amplification():
       # Verify subsampling improves privacy
       assert amplified_epsilon < base_epsilon
   ```

## 🚀 Continuous Deployment

### Release Workflow

**File**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    # Build distribution packages
    
  test-release:
    # Test on Test PyPI
    
  security-final:
    # Final security scan
    
  publish:
    # Publish to PyPI
    
  docker-release:
    # Build and push Docker images
    
  documentation:
    # Deploy documentation
```

### Release Process

1. **Version Bump**
   - Update `pyproject.toml`
   - Update `CHANGELOG.md`
   - Create PR with version changes

2. **Tag Creation**
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

3. **Automated Release**
   - Build packages
   - Run final tests
   - Publish to PyPI
   - Create GitHub release
   - Update documentation

### Deployment Environments

| Environment | Trigger | Purpose |
|-------------|---------|---------|
| Test PyPI | Release tag | Pre-release testing |
| PyPI | Manual approval | Production release |
| Docker Hub | Release tag | Container distribution |
| ReadTheDocs | Push to main | Documentation |

## 📊 Performance Monitoring

### Benchmark Workflow

**File**: `.github/workflows/benchmark.yml`

```yaml
name: Performance Benchmark

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 4 * * 1'  # Weekly benchmarks

jobs:
  cpu-benchmark:
    # CPU performance tests
    
  gpu-benchmark:
    # GPU performance tests (if available)
    
  memory-benchmark:
    # Memory usage profiling
    
  privacy-utility:
    # Privacy-utility curve generation
```

### Benchmark Metrics

- **Training Speed**: Samples per second
- **Memory Usage**: Peak GPU/CPU memory
- **Privacy Cost**: Epsilon consumption rate
- **Accuracy**: Model performance metrics
- **Communication**: Bytes transferred per round

## 🔧 Workflow Configuration

### Required Secrets

Configure these secrets in GitHub repository settings:

```yaml
# PyPI publishing
PYPI_API_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
TEST_PYPI_API_TOKEN: ${{ secrets.TEST_PYPI_API_TOKEN }}

# Docker Hub
DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
DOCKER_PASSWORD: ${{ secrets.DOCKER_PASSWORD }}

# WandB integration
WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}

# Security scanning
SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
```

### Environment Variables

```yaml
# Default test configuration
PYTEST_ARGS: "--maxfail=1 --tb=short"
PRIVACY_TEST_EPSILON: "8.0"
PRIVACY_TEST_DELTA: "1e-5"

# GPU testing
CUDA_VISIBLE_DEVICES: "0"
TORCH_CUDA_ARCH_LIST: "7.0;8.0"
```

## 📈 Workflow Monitoring

### Status Badges

Add these badges to README.md:

```markdown
[![CI](https://github.com/user/repo/workflows/CI/badge.svg)](https://github.com/user/repo/actions/workflows/ci.yml)
[![Security](https://github.com/user/repo/workflows/Security/badge.svg)](https://github.com/user/repo/actions/workflows/security.yml)
[![Privacy](https://github.com/user/repo/workflows/Privacy/badge.svg)](https://github.com/user/repo/actions/workflows/privacy.yml)
```

### Notifications

Configure notifications for:
- **Failed builds** on main branch
- **Security vulnerabilities** discovered
- **Privacy test failures**
- **Deployment completions**

## 🛠️ Local Development

### Pre-commit Integration

Ensure local development matches CI:

```yaml
# .pre-commit-ci.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: ["-x", "--tb=short"]
```

### Local CI Simulation

```bash
# Run full CI locally
make ci

# Individual checks
make lint
make type-check
make security
make test-cov
```

## 🐛 Troubleshooting

### Common Issues

1. **Privacy Test Failures**
   ```bash
   # Check privacy parameters
   python -c "from dp_federated_lora import PrivacyEngine; print(PrivacyEngine.validate_config())"
   ```

2. **GPU Test Failures**
   ```bash
   # Verify CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Import Errors**
   ```bash
   # Check package installation
   pip install -e ".[dev]"
   python -c "import dp_federated_lora; print('OK')"
   ```

### Debug Mode

Enable debug mode for detailed workflow logs:

```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## 📝 Workflow Templates

### Creating New Workflows

1. **Copy existing template**
2. **Update trigger conditions**
3. **Modify job matrix as needed**
4. **Add required secrets**
5. **Test with draft PR**

### Workflow Best Practices

- **Fail fast**: Stop on first error
- **Cache dependencies**: Speed up builds
- **Parallel jobs**: Run tests concurrently
- **Conditional steps**: Skip unnecessary work
- **Clear naming**: Descriptive job and step names

## 📞 Support

For CI/CD issues:
- **GitHub Issues**: Workflow bugs and feature requests
- **Discussions**: Setup questions and best practices
- **Maintainer Review**: Complex workflow changes

---

*This workflow documentation ensures consistent, secure, and privacy-preserving development practices.*