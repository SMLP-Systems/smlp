FROM opensuse/leap:15.5

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
RUN zypper refresh && zypper install -y \
        ca-certificates \
        curl \
        git \
        gpg2 \
        jq \
        libgomp1 \
        glibc-locale \
        glibc-locale-base \
        glibc-i18ndata \
        gzip \
        tar \
        tcsh \
        timezone \
        vim \
        wget \
        xorg-x11-fonts \
        xorg-x11-fonts-core \
        xorg-x11-server-Xvfb \
        xvfb-run \
        xset \
        procps \
        python311 \
        python311-devel \
        python311-tk && \
    zypper clean --all

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
    python3.11 -m pip install --upgrade setuptools

# ---------------------------------------------------------------------------
# 4. Clone smlp (ARG busts the cache to always get latest)
# ---------------------------------------------------------------------------
ARG CACHE_BUST_SMLP
ARG GIT_BRANCH=master
RUN echo "Building image for branch: $GIT_BRANCH"
COPY run_git_clone /app
RUN cd /app && ./run_git_clone $GIT_BRANCH

# ---------------------------------------------------------------------------
# 5. Install smlp from wheel
# ---------------------------------------------------------------------------
RUN python3.11 -m pip install /app/smlp/package_test/python3.11/dist/smlp-0.1.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

# ---------------------------------------------------------------------------
# 6. Configure UTF-8 locale
# ---------------------------------------------------------------------------
RUN localedef -i en_US -f UTF-8 en_US.UTF-8
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
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# ---------------------------------------------------------------------------
# 10. Copy and run DORA
# ---------------------------------------------------------------------------
COPY run_dora /app
RUN ./run_dora

# Default command
CMD ["/usr/bin/bash"]
