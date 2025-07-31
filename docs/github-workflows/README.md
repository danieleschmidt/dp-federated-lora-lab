# GitHub Actions Workflows

This directory contains production-ready GitHub Actions workflows for the dp-federated-lora-lab repository.

## ⚠️ Manual Setup Required

Due to GitHub security restrictions, these workflow files must be manually added to `.github/workflows/` directory by a repository maintainer with appropriate permissions.

## Workflow Files

### 1. `ci.yml` - Comprehensive CI Pipeline
- Multi-platform testing (Linux, macOS, Windows)
- Python version matrix (3.9, 3.10, 3.11)
- Privacy-specific test automation
- ML model validation
- Docker build verification

### 2. `security.yml` - Security Scanning
- Dependency vulnerability scanning
- Static code analysis with bandit
- Secret detection
- Container security scanning
- Privacy compliance checks

### 3. `release.yml` - Automated Release
- Semantic versioning
- Multi-platform Docker builds
- PyPI package publishing  
- Comprehensive validation pipeline

### 4. `dependency-update.yml` - Dependency Management
- Automated security updates
- Compatibility testing
- Automated PR creation for updates

## Setup Instructions

1. **Copy workflow files** from `docs/github-workflows/` to `.github/workflows/`
2. **Configure GitHub secrets** (see documentation in each workflow file)
3. **Test workflows** by creating a test PR
4. **Monitor execution** and adjust configurations as needed

## Required GitHub Secrets

```bash
CODECOV_TOKEN=<coverage-reporting-token>
PYPI_TOKEN=<pypi-publishing-token>
DOCKER_HUB_USERNAME=<docker-hub-username>  
DOCKER_HUB_TOKEN=<docker-hub-access-token>
```

## Security Considerations

All workflows follow security best practices:
- Minimal permissions (principle of least privilege)
- Secret handling with GitHub secrets
- Container security scanning
- Dependency vulnerability checks
- Privacy-aware testing configurations

Once manually installed, these workflows will provide comprehensive CI/CD automation for this ML/Privacy repository.