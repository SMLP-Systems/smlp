FROM ubuntu:22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
        ca-certificates curl gnupg wget \
        gcc g++ git make m4 pkg-config && \
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/deadsnakes.gpg && \
    echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
        > /etc/apt/sources.list.d/deadsnakes.list && \
    apt-get update && apt-get install -y \
        python3.13 python3.13-dev python3.13-venv python3.13-tk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Install pip for Python 3.13
# ---------------------------------------------------------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

# ---------------------------------------------------------------------------
# 3. Python build tools
# ---------------------------------------------------------------------------
RUN python3.13 -m pip install --user --force-reinstall --break-system-packages setuptools==80.10.1 && \
    python3.13 -m pip install --user meson ninja z3-solver==4.8.12 auditwheel patchelf

# ---------------------------------------------------------------------------
# 4. Add ~/.local/bin to PATH
# ---------------------------------------------------------------------------
ENV PATH="/root/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# 5. Clone smlp and optionaly switch to branch
# ---------------------------------------------------------------------------

WORKDIR /app
ARG GIT_BRANCH=smlp_python313
COPY run_git_clone .
RUN ./run_git_clone $GIT_BRANCH

# ---------------------------------------------------------------------------
# 6. Build wheel
# ---------------------------------------------------------------------------
WORKDIR smlp
RUN python3.13 -m pip wheel . -w dist/ && \
    python3.13 repair_wheel.py dist/

# The manylinux wheel is in /app/smlp/dist/
