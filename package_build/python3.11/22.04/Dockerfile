FROM ubuntu:22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
        ca-certificates curl wget \
        gcc g++ git make m4 pkg-config && \
    apt-get update && apt-get install -y \
        python3.11 python3.11-dev python3.11-distutils python3.11-venv python3.11-tk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Install pip for Python 3.11
# ---------------------------------------------------------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ---------------------------------------------------------------------------
# 3. Python build tools
# ---------------------------------------------------------------------------
RUN python3.11 -m pip install --user --force-reinstall --break-system-packages setuptools && \
    python3.11 -m pip install --user meson ninja z3-solver==4.8.12 auditwheel patchelf

# ---------------------------------------------------------------------------
# 4. Add ~/.local/bin to PATH
# ---------------------------------------------------------------------------
ENV PATH="/root/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# 5. Clone smlp and optionaly switch to branch
# ---------------------------------------------------------------------------

WORKDIR /app
ARG GIT_BRANCH=smlp_python311
COPY run_git_clone .
RUN ./run_git_clone $GIT_BRANCH

# ---------------------------------------------------------------------------
# 6. Build wheel
# ---------------------------------------------------------------------------
WORKDIR smlp
RUN python3.11 -m pip wheel . -w dist/ && \
    python3.11 repair_wheel.py dist/

# The manylinux wheel is in /app/smlp/dist/
