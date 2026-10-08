# Isolates neural-trader's Python environment, processes, and file writes
# (logs/, checkpoints/, decisions.jsonl) from anything else on the host.
# See the "Running this in Docker" section of the README for usage.

FROM python:3.11-slim

WORKDIR /app

# Dependencies first, in their own layer, so editing src/ or scripts/ later
# doesn't bust the pip cache and force a full reinstall.
COPY requirements.txt requirements-dev.txt ./

# CPU-only torch wheel -- same reasoning as .github/workflows/tests.yml:
# the default index can pull a CUDA build, which is large and useless here
# (no GPU passthrough). requirements-dev.txt pulls in requirements.txt plus
# pytest, so the image can also run the test suite as a sanity check.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-dev.txt

# Now the project itself. .dockerignore keeps .git, .env, logs/,
# checkpoints/, __pycache__, and any local venv out of the build context.
COPY . .

# These are normally bind-mounted (see docker-compose.yml) so the data
# survives a container restart and stays visible on the host, but create
# them here too so the image is self-contained if run without volumes.
RUN mkdir -p logs checkpoints

# Run as a non-root user rather than root inside the container.
RUN useradd --create-home --uid 1000 trader && chown -R trader:trader /app
USER trader

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8787

# Default service: the dashboard. The live trading loop is run as a
# separate container from the same image -- see docker-compose.yml, or
# override this at `docker run` time with:
#   docker run ... neural-trader python scripts/run_live_alpaca.py
CMD ["python", "scripts/serve_dashboard.py", "--host", "0.0.0.0", "--port", "8787"]
