# Multi-stage Docker build for dp-federated-lora-lab
# Optimized for production deployment with security best practices

# Base image with Python and CUDA support
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Metadata
LABEL maintainer="daniel@terragonlabs.com"
LABEL description="Differential Privacy Federated LoRA Lab - Production Container"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/yourusername/dp-federated-lora-lab"

# Build arguments
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.1.0
ARG CUDA_VERSION=12.1
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Security: Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential system packages
    build-essential \
    software-properties-common \
    curl \
    wget \
    git \
    ca-certificates \
    # Python dependencies
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    # Security and monitoring
    openssh-client \
    gnupg2 \
    # Cleanup
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Create symbolic links for Python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip and install essential packages
RUN python -m pip install --upgrade pip setuptools wheel

# Development stage
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    tree \
    jupyter \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Set working directory
WORKDIR /workspace

# Copy source code
COPY . /workspace/

# Install package in development mode
RUN pip install -e ".[dev,docs,benchmark]"

# Switch to non-root user
USER appuser

# Expose ports for development
EXPOSE 8000 8888 6006

# Development entrypoint
CMD ["bash"]

# Production stage
FROM base as production

# Install only production system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Minimal runtime dependencies
    libgomp1 \
    libffi-dev \
    libssl-dev \
    # Cleanup
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean up pip cache and temporary files
    pip cache purge && \
    rm -rf /tmp/* /var/tmp/* /root/.cache

# Copy application source
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./

# Install the package
RUN pip install --no-cache-dir -e . && \
    # Remove build dependencies
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    # Final cleanup
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/outputs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import dp_federated_lora; print('Health check passed')" || exit 1

# Expose application port
EXPOSE 8080

# Set entrypoint and default command
ENTRYPOINT ["python", "-m", "dp_federated_lora.cli"]
CMD ["--help"]

# Production runtime stage (minimal)
FROM production as runtime

# Metadata for production image
LABEL build_date=${BUILD_DATE}
LABEL vcs_ref=${VCS_REF}
LABEL version=${VERSION}

# Final security hardening
USER appuser

# Create volume mount points
VOLUME ["/app/data", "/app/models", "/app/outputs"]

# Environment variables for production
ENV PYTHONPATH=/app \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    MAX_WORKERS=4

# GPU-enabled stage
FROM runtime as gpu

# Switch back to root for GPU setup
USER root

# Install additional GPU dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvidia-compute-470 \
    libnvidia-gl-470 \
    && rm -rf /var/lib/apt/lists/*

# Switch back to appuser
USER appuser

# GPU-specific environment variables
ENV CUDA_CACHE_PATH=/tmp/cuda_cache \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6" \
    FORCE_CUDA=1

# CPU-only stage (for environments without GPU)
FROM runtime as cpu

# Override GPU environment variables
ENV CUDA_VISIBLE_DEVICES="" \
    NVIDIA_VISIBLE_DEVICES="" \
    TORCH_CUDA_ARCH_LIST=""

# Install CPU-only PyTorch if needed (uncomment if you have CPU-specific requirements)
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Benchmark stage (for performance testing)
FROM development as benchmark

# Install additional benchmarking tools
RUN pip install --no-cache-dir \
    memory-profiler \
    line-profiler \
    py-spy \
    psutil

# Copy benchmark scripts
COPY benchmarks/ ./benchmarks/
COPY examples/ ./examples/

# Benchmark entrypoint
ENTRYPOINT ["python", "-m", "dp_federated_lora.benchmarks.run_all"]

# Default target
FROM runtime as final

# Final metadata
LABEL final_stage="runtime"
LABEL gpu_support="optional"
LABEL security_hardened="true"