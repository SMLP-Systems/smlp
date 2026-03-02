FROM ubuntu:22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
        ca-certificates \
        curl \
        git \
        gnupg \
        jq \
        libgomp1 \
        locales \
        tzdata wget \
        xvfb && \
    apt-get update && apt-get install -y \
        python3.11 python3.11-dev python3.11-distutils python3.11-venv python3.11-tk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------------------------------
# 2. mathsat
# ---------------------------------------------------------------------------
WORKDIR /app
COPY run_mathsat_build.sh .
RUN ./run_mathsat_build.sh && rm -rf /tmp/mathsat*

# ---------------------------------------------------------------------------
# 3. Install pip and setuptools for Python 3.11
# ---------------------------------------------------------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    python3.11 -m pip install --user --force-reinstall --break-system-packages setuptools

# ---------------------------------------------------------------------------
# 4. Clone smlp (ARG busts the cache to always get latest)
# ---------------------------------------------------------------------------
ARG CACHE_BUST_SMLP
COPY run_git_clone /app
RUN cd /app && ./run_git_clone smlp_python311

# ---------------------------------------------------------------------------
# 5. Install smlp from wheel
# ---------------------------------------------------------------------------
RUN python3.11 -m pip install /app/smlp/dist/smlp-0.1.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

# ---------------------------------------------------------------------------
# 6. Install UTF-8 fonts
# ---------------------------------------------------------------------------
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# ---------------------------------------------------------------------------
# 7. Docker GUI patch
# ---------------------------------------------------------------------------
COPY tkagg_patch.sh .
RUN ./tkagg_patch.sh 

# ---------------------------------------------------------------------------
# 8. SMLP regression patch
# ---------------------------------------------------------------------------
COPY smlp_regr_patch.sh .
RUN ./smlp_regr_patch.sh

# ---------------------------------------------------------------------------
# 9. Define python version
# ---------------------------------------------------------------------------
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# ---------------------------------------------------------------------------
# 11. Copy and run DORA 
# ---------------------------------------------------------------------------
COPY run_dora /app
RUN ./run_dora

# Default command
CMD ["/usr/bin/bash"]
