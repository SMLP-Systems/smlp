FROM ubuntu:24.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
        git \
        jq \
        libgomp1 \
        locales \
        tzdata wget \
        xvfb && \
    apt-get update && apt-get install -y \
        python3 python3-dev python3-venv python3-tk python3-pip python3-setuptools python-is-python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------------------------------
# 2. mathsat
# ---------------------------------------------------------------------------
WORKDIR /app
COPY run_mathsat_build.sh .
RUN ./run_mathsat_build.sh && rm -rf /tmp/mathsat*

# ---------------------------------------------------------------------------
# 3. Clone smlp (ARG busts the cache to always get latest)
# ---------------------------------------------------------------------------
ARG CACHE_BUST_SMLP
COPY run_git_clone /app
RUN cd /app && ./run_git_clone smlp_python312

# ---------------------------------------------------------------------------
# 4. Install smlp from wheel
# ---------------------------------------------------------------------------
RUN python3 -m pip install /app/smlp/dist/smlp-0.1.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl --target /usr/local/lib/python3.12/dist-packages

# ---------------------------------------------------------------------------
# 5. Install pycaret from archive
# ---------------------------------------------------------------------------
RUN pip install /app/smlp/docker/python3.13/pycaret_313.tar.gz --upgrade --target /usr/local/lib/python3.12/dist-packages

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
COPY ./tkagg_patch.sh .
RUN ./tkagg_patch.sh 

# ---------------------------------------------------------------------------
# 8. SMLP regression patch
# ---------------------------------------------------------------------------
COPY ./smlp_regr_patch.sh .
RUN ./smlp_regr_patch.sh

# ---------------------------------------------------------------------------
# 9. Copy and run DORA 
# ---------------------------------------------------------------------------
COPY ./run_dora /app
RUN ./run_dora

# Default command
CMD ["/usr/bin/bash"]
