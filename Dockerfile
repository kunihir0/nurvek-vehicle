# Dockerfile for Nurvek LP Trainer with ROCm
ARG ROCM_VERSION=6.3.3
FROM rocm/dev-almalinux-8:${ROCM_VERSION}-complete AS base

# Install system dependencies
RUN yum update -y && \
    yum install -y git curl gcc make && \
    # gcc and make might be needed for building Rust/Cargo or uv from source
    yum clean all && \
    rm -rf /var/cache/yum

# Install Rust and Cargo
# This will install them to /root/.cargo/bin by default when run as root
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path

# Install uv using cargo
# This should also install uv to /root/.cargo/bin
RUN cargo install --git https://github.com/astral-sh/uv uv

# Verify uv installation by checking its version
RUN uv --version

# Set working directory
WORKDIR /app

# Copy project files
# .dockerignore should prevent large/unnecessary files from being copied
COPY . .

# Create a virtual environment and install Python dependencies using uv
# Ensure Python 3.9 is used. The base rocm/dev-almalinux-8 image provides Python 3.9.
RUN uv venv .venv --python python3.9 && \
    # The following activation is for this RUN layer only.
    # uv run will handle the environment for the CMD.
    . .venv/bin/activate && \
    uv pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3 && \
    uv pip install --no-cache-dir ultralytics pyyaml

# Set environment variables for ROCm
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0

# Default command to run the training script using uv run
CMD ["uv", "run", "./train_lp_detector.py"]
