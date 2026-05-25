FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=0
ENV UV_PROJECT_ENVIRONMENT="/venv"
ENV PATH="/venv/bin:$PATH"

# Run system updates and install required X11/GL libraries for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1
# Prevent uv from using hardlinks (safer in Docker)
ENV UV_LINK_MODE=copy

# Step 1: Copy only dependency files to cache the installation layer
COPY pyproject.toml uv.lock ./

# Step 2: Install dependencies (creates .venv inside /app)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Step 3: Copy the actual application code
COPY . .

# Step 4: Final sync to install the project itself
# RUN uv sync --no-dev

# Put the virtual environment on the PATH
# ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
