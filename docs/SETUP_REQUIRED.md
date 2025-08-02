# 🚀 Manual Setup Required

This document outlines the manual setup steps required to complete the SDLC implementation due to GitHub App permission limitations.

## ⚠️ Required Actions

### 1. GitHub Actions Workflows

Due to GitHub security restrictions, the following workflow files must be manually copied from `docs/github-workflows/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy essential workflows
cp docs/github-workflows/ci.yml .github/workflows/
cp docs/github-workflows/security.yml .github/workflows/
cp docs/github-workflows/release.yml .github/workflows/
cp docs/github-workflows/dependency-update.yml .github/workflows/

# Optional: Copy autonomous workflows for enhanced SDLC
cp docs/github-workflows/autonomous-ci.yml .github/workflows/ci.yml
cp docs/github-workflows/autonomous-security.yml .github/workflows/security.yml
```

### 2. GitHub Secrets Configuration

Configure the following secrets in your GitHub repository settings:

#### Required Secrets
```bash
# Code coverage reporting
CODECOV_TOKEN=<your-codecov-token>

# Package publishing
PYPI_TOKEN=<your-pypi-token>

# Database credentials (for CI/CD)
POSTGRES_PASSWORD=<secure-password>
```

#### Optional Secrets (for enhanced features)
```bash
# Docker Hub publishing
DOCKER_HUB_USERNAME=<your-docker-username>
DOCKER_HUB_TOKEN=<your-docker-token>

# Slack notifications
SLACK_WEBHOOK_URL=<your-slack-webhook>

# AWS/Cloud deployment (if using)
AWS_ACCESS_KEY_ID=<your-aws-key>
AWS_SECRET_ACCESS_KEY=<your-aws-secret>

# Security scanning enhancements
SNYK_TOKEN=<your-snyk-token>
```

### 3. Branch Protection Rules

Configure branch protection for the `main` branch with these settings:

#### Via GitHub UI (Settings → Branches)
- ✅ Require a pull request before merging
- ✅ Require approvals (minimum 1)
- ✅ Dismiss stale PR approvals when new commits are pushed
- ✅ Require review from CODEOWNERS
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require conversation resolution before merging
- ✅ Include administrators

#### Required Status Checks
- `quality` (from CI workflow)
- `test` (from CI workflow)
- `privacy-tests` (from CI workflow)
- `security` (from security workflow)

### 4. Repository Settings

#### General Settings
- ✅ Enable Issues
- ✅ Enable Projects
- ✅ Enable Wiki
- ✅ Enable Discussions (recommended)
- ✅ Enable Sponsorships (if applicable)

#### Security Settings
- ✅ Enable private vulnerability reporting
- ✅ Enable Dependabot alerts
- ✅ Enable Dependabot security updates
- ✅ Enable Dependabot version updates
- ✅ Enable secret scanning
- ✅ Enable push protection for secrets

#### Code Security Analysis
- ✅ Enable CodeQL analysis
- ✅ Setup SARIF uploads from workflows

### 5. Third-Party Integrations

#### Codecov Integration
1. Sign up at [codecov.io](https://codecov.io)
2. Connect your GitHub repository
3. Add `CODECOV_TOKEN` to repository secrets

#### PyPI Publishing (if applicable)
1. Create account at [pypi.org](https://pypi.org)
2. Generate API token
3. Add `PYPI_TOKEN` to repository secrets

### 6. Environment Files

Create the following environment files for local development:

#### `.env.example` → `.env`
```bash
# Copy and customize
cp .env.example .env

# Edit with your local configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=dp_federated_lora
POSTGRES_USER=dpuser
POSTGRES_PASSWORD=your-local-password

REDIS_HOST=localhost
REDIS_PORT=6379

# Privacy settings
PRIVACY_BUDGET_EPSILON=8.0
PRIVACY_BUDGET_DELTA=1e-5

# ML settings  
MODEL_CACHE_DIR=./models
DATA_CACHE_DIR=./data

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Development
DEBUG=true
LOG_LEVEL=DEBUG
```

### 7. Database Setup

#### Local PostgreSQL
```sql
-- Connect to PostgreSQL and run:
CREATE DATABASE dp_federated_lora;
CREATE USER dpuser WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE dp_federated_lora TO dpuser;
```

#### Docker Setup (Alternative)
```bash
# Use docker-compose for easy setup
docker-compose up -d postgres redis

# Wait for services to start
docker-compose ps
```

### 8. Pre-commit Hooks

Install pre-commit hooks for code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

### 9. Documentation Website (Optional)

If you want to deploy documentation:

#### GitHub Pages
1. Go to Settings → Pages
2. Select source: "Deploy from a branch"
3. Branch: `gh-pages` (will be created by workflow)
4. Folder: `/` (root)

#### Custom Domain (Optional)
1. Add `CNAME` file with your domain
2. Configure DNS with your provider
3. Enable HTTPS in GitHub Pages settings

### 10. Monitoring Setup

#### Grafana Dashboards
1. Access Grafana at `http://localhost:3000`
2. Default login: `admin/admin`
3. Import dashboards from `monitoring/grafana/dashboards/`
4. Configure data sources:
   - Prometheus: `http://prometheus:9090`

#### Prometheus Configuration
1. Review `monitoring/prometheus.yml`
2. Customize scrape targets for your environment
3. Add alerting rules in `monitoring/rules/`

## 🎯 Verification Checklist

After completing manual setup, verify everything works:

### ✅ Basic Functionality
- [ ] Repository clones successfully
- [ ] `make install-dev` completes without errors
- [ ] `make test-fast` passes
- [ ] `make lint` passes
- [ ] `make type-check` passes
- [ ] `make security` passes

### ✅ CI/CD Pipeline
- [ ] Pull requests trigger CI workflows
- [ ] All workflow jobs complete successfully
- [ ] Security scans run and report results
- [ ] Test coverage is reported
- [ ] Artifacts are uploaded

### ✅ Branch Protection
- [ ] Direct pushes to `main` are blocked
- [ ] Pull requests require reviews
- [ ] Status checks must pass before merge
- [ ] CODEOWNERS reviews are required

### ✅ Integrations
- [ ] Code coverage reports to Codecov
- [ ] Security alerts are enabled
- [ ] Dependabot creates update PRs
- [ ] Secrets scanning is active

### ✅ Development Environment
- [ ] Docker containers start successfully
- [ ] Database connections work
- [ ] Redis is accessible
- [ ] Health checks pass
- [ ] Monitoring dashboards load

### ✅ Automation
- [ ] `make metrics-collect` runs successfully
- [ ] `make metrics-report` generates reports
- [ ] Pre-commit hooks run on commits
- [ ] Dependency updates are automated

## 🚨 Security Considerations

### Repository Secrets
- Never commit secrets to the repository
- Use GitHub Secrets for sensitive configuration
- Rotate secrets regularly
- Review secret access permissions

### Access Control
- Limit repository access to necessary team members
- Use teams for permission management
- Enable two-factor authentication
- Review audit logs regularly

### Dependency Management
- Keep dependencies up to date
- Review Dependabot PRs promptly
- Monitor security advisories
- Use lock files for reproducible builds

## 📞 Support

If you encounter issues during setup:

1. **Check the logs**: Review workflow run logs for detailed error messages
2. **Review documentation**: Check relevant documentation for each component
3. **Search issues**: Look for similar issues in the repository
4. **Ask for help**: Create an issue with detailed information about the problem

## 🎉 Completion

Once all manual setup steps are completed, your repository will have:

- ✅ Full CI/CD pipeline with comprehensive testing
- ✅ Automated security scanning and monitoring
- ✅ Comprehensive metrics collection and reporting
- ✅ Development environment with Docker support
- ✅ Documentation website and API docs
- ✅ Automated dependency management
- ✅ Advanced monitoring and observability
- ✅ Privacy-preserving ML development tools

Welcome to your fully-featured SDLC! 🚀